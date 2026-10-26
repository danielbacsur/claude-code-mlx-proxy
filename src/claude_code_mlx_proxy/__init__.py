import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# Address the proxy listens on, and the default address of the model server.
HOST = "127.0.0.1"

# Where mlx_lm.server answers; main resets it from the server's --host and --port.
UPSTREAM = urlsplit(f"http://{HOST}:8080")

# mlx_lm.server's name for the model it started with, so it never loads another.
MODEL = "default_model"

# Port the proxy listens on when standalone; wrapping a command picks a free one.
PORT = 8082

# finish_reason tool_calls maps to tool_use only if a call survived mlx's parser.
STOP = {"length": "max_tokens"}

# Model the wrapper starts mlx_lm.server with when the user passes no --model flag.
DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

# File for the output of mlx_lm.server and, while wrapping a command, the proxy log.
LOG_PATH = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
    "claude-code-mlx-proxy.log",
)

# Where the proxy writes its access log; main points it at LOG_PATH when wrapping.
LOG = sys.stderr

# Anthropic error types for upstream status codes; any other code becomes api_error.
ERRORS = {
    400: "invalid_request_error",
    411: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
}


# Flattens Anthropic content, either a plain string or a list of content blocks, into
# plain text, keeping text blocks and replacing every image with a placeholder.
def text_of(content):
    if isinstance(content, str):
        return content

    parts = []

    for b in content or []:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "image":
            parts.append("[image omitted]")

    return "\n".join(parts)


# Translates an Anthropic Messages request into an OpenAI chat completion request for
# mlx_lm.server, covering the system prompt, messages, tool definitions and options.
def to_openai(req):
    msgs = []
    system = text_of(req.get("system") or "")

    # Claude Code's per-session billing header line would defeat prefix caching.
    if system.startswith("x-anthropic-billing-header"):
        system = system.partition("\n")[2]

    if system:
        msgs.append({"role": "system", "content": system})

    for m in req.get("messages", []):
        # Chat templates drop non-leading system turns; Claude Code puts cwd in one.
        if m.get("role") == "system":
            m = dict(m, role="user")

        content = m.get("content")

        if isinstance(content, str):
            msgs.append({"role": m["role"], "content": content})
            continue

        if m["role"] == "assistant":
            text, calls = [], []

            for b in content:
                if b.get("type") == "text":
                    text.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    fn = {
                        "name": b["name"],
                        "arguments": json.dumps(b.get("input", {})),
                    }

                    calls.append({"id": b["id"], "type": "function", "function": fn})

            msg = {"role": "assistant", "content": "".join(text)}

            if calls:
                msg["tool_calls"] = calls

            msgs.append(msg)
            continue

        # Tool results go first so they directly follow the assistant tool_calls.
        rest = []

        for b in content:
            if b.get("type") == "tool_result":
                out = text_of(b.get("content"))

                if b.get("is_error"):
                    out = "Error: " + out

                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id"),
                        "content": out,
                    }
                )
            else:
                rest.append(b)

        if rest:
            msgs.append({"role": m["role"], "content": text_of(rest)})

    body = {"model": MODEL, "messages": msgs, "stream": bool(req.get("stream"))}

    if body["stream"]:
        body["stream_options"] = {"include_usage": True}

    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        if key in req:
            body[key] = req[key]

    if req.get("stop_sequences"):
        body["stop"] = req["stop_sequences"]

    choice = req.get("tool_choice") or {}

    if req.get("tools") and choice.get("type") != "none":
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in req["tools"]
            if "name" in t
        ]

        if choice.get("type") == "any":
            body["tool_choice"] = "required"
        elif choice.get("type") == "tool":
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": choice.get("name")},
            }

    return body


# Turns a complete OpenAI chat completion into an Anthropic message, mapping reasoning,
# text and tool calls to content blocks and the finish reason to a stop reason.
def to_anthropic(resp, model):
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []

    if msg.get("reasoning"):
        content.append(
            {"type": "thinking", "thinking": msg["reasoning"], "signature": "local"}
        )

    if (msg.get("content") or "").strip():
        content.append({"type": "text", "text": msg["content"].lstrip()})

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})

        content.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or new_id("toolu"),
                "name": fn.get("name"),
                "input": parse_args(fn.get("arguments")),
            }
        )

    usage = resp.get("usage") or {}

    stop = (
        "tool_use"
        if msg.get("tool_calls")
        else STOP.get(choice.get("finish_reason"), "end_turn")
    )

    return {
        "id": new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# Parses tool call arguments from the JSON string the model produced into a dict,
# keeping the raw text under a raw key whenever the arguments are not valid JSON.
def parse_args(args):
    if isinstance(args, dict):
        return args

    try:
        return json.loads(args or "{}")
    except ValueError:
        return {"raw": args}


# Builds a random identifier with an Anthropic style prefix such as msg or toolu, for
# messages and for tool calls that the upstream server returned without any id.
def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# Roughly estimates the input tokens of a request from the size of its JSON, for
# count_tokens and for message_start before mlx_lm.server reports the real counts.
def estimate_tokens(req):
    size = len(json.dumps([req.get("system"), req.get("messages"), req.get("tools")]))
    return max(1, size // 4)


class Stream:
    # Starts an Anthropic event stream by sending message_start, then keeps track of the
    # open content block, its index, the token usage so far and the stop reason.
    def __init__(self, send, model, input_tokens):
        self.send, self.index, self.kind = send, -1, None
        self.usage = {"input_tokens": input_tokens, "output_tokens": 0}
        self.stop = "end_turn"
        self.tool_seen = False

        msg = {
            "id": new_id("msg"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": dict(self.usage),
        }

        send("message_start", {"type": "message_start", "message": msg})

    # Closes the current content block if one is open, then opens a new block of the
    # given kind at the next index and sends its content_block_start event right away.
    def open(self, kind, block):
        self.close()
        self.index += 1
        self.kind = kind

        self.send(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.index,
                "content_block": block,
            },
        )

    # Ends the open content block, first signing it off with a placeholder signature
    # when it is a thinking block, and does nothing when no content block is open.
    def close(self):
        if self.kind is None:
            return

        if self.kind == "thinking":
            self.delta({"type": "signature_delta", "signature": "local"})

        self.send(
            "content_block_stop", {"type": "content_block_stop", "index": self.index}
        )

        self.kind = None

    # Sends one content_block_delta event for the open block, carrying a piece of text,
    # of thinking or of partial tool input JSON as the payload the caller passes in.
    def delta(self, d):
        self.send(
            "content_block_delta",
            {"type": "content_block_delta", "index": self.index, "delta": d},
        )

    # Translates one upstream chunk into block events, giving reasoning, tool calls and
    # text their own content blocks, and records the usage and finish reason it carries.
    def chunk(self, data):
        if data.get("usage"):
            self.usage["input_tokens"] = data["usage"].get(
                "prompt_tokens", self.usage["input_tokens"]
            )

            self.usage["output_tokens"] = data["usage"].get("completion_tokens", 0)

        for choice in data.get("choices") or []:
            d = choice.get("delta") or {}

            if d.get("reasoning"):
                if self.kind != "thinking":
                    self.open("thinking", {"type": "thinking", "thinking": ""})

                self.delta({"type": "thinking_delta", "thinking": d["reasoning"]})

            # A delta with both holds a tool call and the text generated after it.
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") or {}

                # A new id (or name) starts a new tool block; later fragments extend it.
                if tc.get("id") or fn.get("name") or self.kind != "tool":
                    self.tool_seen = True

                    block = {
                        "type": "tool_use",
                        "id": tc.get("id") or new_id("toolu"),
                        "name": fn.get("name", ""),
                        "input": {},
                    }

                    self.open("tool", block)

                args = fn.get("arguments")

                if args:
                    partial = args if isinstance(args, str) else json.dumps(args)
                    self.delta({"type": "input_json_delta", "partial_json": partial})

            # Whitespace, e.g. between </think> and <tool_call>, never opens a block.
            text = d.get("content") or ""

            if self.kind != "text":
                text = text.lstrip()

                if text:
                    self.open("text", {"type": "text", "text": ""})

            if text:
                self.delta({"type": "text_delta", "text": text})

            if choice.get("finish_reason"):
                self.stop = STOP.get(choice["finish_reason"], "end_turn")

    # Closes the last open block and ends the message with its stop reason and the final
    # token usage, reporting tool_use as the reason whenever any tool call came through.
    def finish(self):
        self.close()
        stop = "tool_use" if self.tool_seen else self.stop

        self.send(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": self.usage,
            },
        )

        self.send("message_stop", {"type": "message_stop"})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Writes one access log line for every request the proxy handles, to stderr when it
    # runs standalone and to the shared log file while it is wrapping another command.
    def log_message(self, fmt, *args):
        LOG.write("proxy: " + fmt % args + "\n")
        LOG.flush()

    # Sends a complete JSON response with the given status code, setting the content
    # type and the content length so that the client connection can be kept alive.
    def reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # Sends an Anthropic style error response, choosing the error type from the status
    # code and falling back to api_error for any status code that it does not know.
    def error(self, status, message):
        kind = ERRORS.get(status, "api_error")

        self.reply(
            status, {"type": "error", "error": {"type": kind, "message": message}}
        )

    # Answers the model list and the health checks with small JSON documents, and
    # replies with a not found error for any other path a client tries to read.
    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/v1/models":
            self.reply(
                200,
                {
                    "data": [{"type": "model", "id": MODEL, "display_name": MODEL}],
                    "has_more": False,
                },
            )
        elif path in ("/", "/health"):
            self.reply(200, {"status": "ok"})
        else:
            self.error(404, f"unknown endpoint {path}")

    # Answers the HEAD probes that Claude Code sends to check its connection with an
    # empty success response, whichever path on the proxy the probe happens to ask for.
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # Reads and validates the JSON request body, then serves token counting itself or
    # relays a message request upstream, quietly ignoring clients that disconnect early.
    def do_POST(self):
        path = self.path.split("?")[0]
        size = self.headers.get("Content-Length", "")

        if not size.isdigit():
            self.close_connection = True
            return self.error(411, "a request body needs a valid Content-Length")

        try:
            req = json.loads(self.rfile.read(int(size)) or b"{}")
        except ValueError as e:
            return self.error(400, f"invalid JSON: {e}")

        if not isinstance(req, dict):
            return self.error(400, "request body must be a JSON object")

        if path == "/v1/messages/count_tokens":
            return self.reply(200, {"input_tokens": estimate_tokens(req)})

        if path != "/v1/messages":
            return self.error(404, f"unknown endpoint {path}")

        try:
            self.relay(req)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # Sends a translated request to mlx_lm.server and returns its answer as an Anthropic
    # message or event stream, turning any upstream failure into an Anthropic error.
    def relay(self, req):
        model = req.get("model") or MODEL

        try:
            body = to_openai(req)
        except (KeyError, TypeError, AttributeError) as e:
            return self.error(400, f"could not translate request: {e!r}")

        conn = http.client.HTTPConnection(
            UPSTREAM.hostname, UPSTREAM.port or 80, timeout=600
        )

        try:
            try:
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    json.dumps(body),
                    {"Content-Type": "application/json"},
                )

                resp = conn.getresponse()

                if resp.status == 200 and not body["stream"]:
                    msg = to_anthropic(json.loads(resp.read()), model)
            except (
                OSError,
                ValueError,
                AttributeError,
                TypeError,
                http.client.HTTPException,
            ) as e:
                return self.error(502, f"upstream failed: {e!r}")

            if resp.status != 200:
                raw = resp.read().decode(errors="replace")

                try:
                    raw = json.loads(raw).get("error", raw)
                except (ValueError, AttributeError):
                    pass

                msg = (
                    raw.get("message", str(raw)) if isinstance(raw, dict) else str(raw)
                )

                return self.error(resp.status, f"upstream: {msg}")

            if not body["stream"]:
                return self.reply(200, msg)

            self.stream(resp, Stream(self.sse_start(), model, estimate_tokens(req)))
        finally:
            conn.close()

    # Sends the response headers for a server-sent event stream and returns a function
    # that writes one named event with its JSON data to the client and then flushes it.
    def sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        # Writes a single server-sent event to the client and flushes it at once, so
        # that each delta reaches Claude Code as soon as the model has produced it.
        def send(event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        return send

    # Relays the upstream event stream line by line as Anthropic events, and ends the
    # response with an error event whenever the stream breaks before it has finished.
    def stream(self, resp, out):
        try:
            for line in resp:
                line = line.strip()

                # mlx sends keepalive comments while it prefills a prompt; answering
                # each one with a ping lets the proxy notice a client that has gone.
                if line.startswith(b":"):
                    out.send("ping", {"type": "ping"})

                if not line.startswith(b"data:"):
                    continue

                payload = line[5:].strip()

                if payload == b"[DONE]":
                    return out.finish()

                data = json.loads(payload)

                if "error" in data:
                    err = data["error"]
                    message = err.get("message") if isinstance(err, dict) else None
                    raise RuntimeError(message or str(err))

                out.chunk(data)

            # mlx_lm.server just closes the socket when generation dies mid-stream.
            raise RuntimeError("upstream closed the stream early")
        except (
            OSError,
            ValueError,
            RuntimeError,
            AttributeError,
            TypeError,
            http.client.HTTPException,
        ) as e:
            # If the client is gone this write fails too, and do_POST ends it.
            out.send(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": f"upstream: {e}"},
                },
            )


# Prints a status line for the person running the wrapper to stderr, with a fixed prefix
# so that it stands apart from everything that the wrapped command itself prints.
def say(msg):
    print(f"claude-code-mlx-proxy: {msg}", file=sys.stderr, flush=True)


# Finds the value of a flag given as --name value or as --name=value in the argument
# list, and can also remove that flag from the list once its value has been read.
def flag(args, name, remove=False):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            value = args[i + 1]

            if remove:
                del args[i : i + 2]

            return value

        if a.startswith(name + "="):
            if remove:
                del args[i]

            return a.split("=", 1)[1]

    return None


# Checks whether a model server already answers at the upstream address by asking for
# its model list, using a short timeout so that the wrapper still starts quickly.
def upstream_alive():
    conn = http.client.HTTPConnection(UPSTREAM.hostname, UPSTREAM.port or 80, timeout=2)

    try:
        conn.request("GET", "/v1/models")
        return conn.getresponse().status == 200
    except OSError:
        return False
    finally:
        conn.close()


# Starts the bundled mlx_lm.server with the user's arguments and the default model when
# none is given, in a session of its own and with all of its output going to the log.
def start_server(args, log):
    if flag(args, "--model") is None:
        args += ["--model", DEFAULT_MODEL]

    say(f"starting mlx_lm.server with {flag(args, '--model')}, log in {LOG_PATH}")

    # Its own session, so a Ctrl-C meant for the wrapped program never reaches it.
    return subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", *args],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


# Waits until the freshly started model server answers its first request, and exits with
# a pointer to the log file when the server process dies before it becomes ready.
def wait_ready(proc):
    while not upstream_alive():
        if proc.poll() is not None:
            sys.exit(
                f"claude-code-mlx-proxy: mlx_lm.server exited with code "
                f"{proc.returncode}, see {LOG_PATH}"
            )

        time.sleep(0.5)


# Stops a model server that this process started itself, asking it to terminate first
# and killing it outright when it is still running after ten seconds of waiting.
def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()

        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


# Finds the local folder of a model, which is either a path the user gave directly or
# the current snapshot of a Hugging Face repository in the hub cache on this machine.
def model_dir(model):
    if os.path.isdir(model):
        return model

    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(home, "hub")
    repo = os.path.join(hub, "models--" + model.replace("/", "--"))

    try:
        with open(os.path.join(repo, "refs", "main")) as f:
            return os.path.join(repo, "snapshots", f.read().strip())
    except OSError:
        return None


# Reads the trained context window from the model's config.json, so Claude Code knows
# the real limit of the model instead of warning about one it does not recognize.
def context_window(model):
    try:
        with open(os.path.join(model_dir(model), "config.json")) as f:
            cfg = json.load(f)
    except (OSError, TypeError, ValueError):
        return None

    return cfg.get("text_config", cfg).get("max_position_embeddings")


# Runs the wrapped command against a proxy on a free port, with the base URL, the model
# name and the context window set in its environment, and returns its exit code.
def run(cmd, model):
    server = ThreadingHTTPServer((HOST, 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://{HOST}:{server.server_address[1]}"
    env = dict(os.environ, ANTHROPIC_BASE_URL=url)

    if model:
        env.setdefault("ANTHROPIC_MODEL", model)

    # A context window the user set beats the one from the model's config.
    if model and (window := context_window(model)):
        env.setdefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(window))

    context = env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "unknown")
    say(f"running {cmd[0]} on {model or 'the running model'}, context {context}")

    try:
        child = subprocess.Popen(cmd, env=env)
    except OSError as e:
        say(f"cannot run {cmd[0]}: {e.strerror}")
        return 127

    # The terminal sends Ctrl-C to the child too; the wrapper must not die of it.
    signal.signal(signal.SIGINT, lambda *a: None)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda s, f: child.send_signal(s))

    code = child.wait()
    server.shutdown()

    return code if code >= 0 else 128 - code


# Splits the arguments at --, makes sure a model server is running and starts one if
# needed, then either wraps the given command or keeps serving as a standalone proxy.
def main():
    global UPSTREAM, LOG

    args, cmd = sys.argv[1:], []

    if "--" in args:
        i = args.index("--")
        args, cmd = args[:i], args[i + 1 :]

    # The server lives wherever its own --host and --port flags put it.
    host = flag(args, "--host") or HOST
    UPSTREAM = urlsplit(f"http://{host}:{flag(args, '--port') or 8080}")

    # kill or a closed terminal must still stop a server we started.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *a: sys.exit(0))

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    proc = None

    with open(LOG_PATH, "w") as log:
        try:
            if upstream_alive():
                unused = list(args)

                for name in ("--host", "--port"):
                    flag(unused, name, remove=True)

                if unused:
                    say(
                        f"using the server at {UPSTREAM.geturl()}, "
                        f"ignoring: {' '.join(unused)}"
                    )
            else:
                proc = start_server(args, log)
                wait_ready(proc)

            model = flag(args, "--model")

            if cmd:
                LOG = log
                sys.exit(run(cmd, model))

            server = ThreadingHTTPServer((HOST, PORT), Handler)
            server.daemon_threads = True

            say(f"proxy on http://{HOST}:{PORT} -> {UPSTREAM.geturl()}")

            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop_server(proc)


if __name__ == "__main__":
    main()
