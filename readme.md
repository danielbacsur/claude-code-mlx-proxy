<h1 align="center">Claude Code MLX Proxy</h1>

<p align="center">
  A local proxy that brings MLX models into Claude Code.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/danielbacsur/claude-code-mlx-proxy/main/readme.png" alt="Claude Code MLX Proxy" width="80%" />
</p>

<br/>

<h2>Installation</h2>

```bash
uv tool install git+https://github.com/danielbacsur/claude-code-mlx-proxy
```

<div align="justify">
  <p>
    <b>All you need is an Apple Silicon Mac with <a href="https://docs.astral.sh/uv/">uv</a> and Claude Code.</b> The install brings its own copy of <code>mlx-lm</code>, so there is no separate model server to set up, and the first run downloads the default model, <code>Qwen3.5-4B</code>, which is about 3 GB.
  </p>
</div>

<br/>

<h2>Usage</h2>

```bash
claude-code-mlx-proxy -- claude
```

<div align="justify">
  <p>
    <b>Put the proxy in front of <code>claude</code> and that is the whole setup.</b> It starts <code>mlx_lm.server</code>, starts the proxy, and runs Claude Code with its environment already pointed at them, showing the real model name and context window in the header. When Claude Code exits, everything it started is stopped again &mdash; a server you were already running yourself is reused and left alone. Anything after <code>--</code> is passed to Claude Code untouched, and anything before it goes straight to <code>mlx_lm.server</code>, so every option of either one works without the proxy knowing about it.
  </p>
</div>

```bash
claude-code-mlx-proxy --model mlx-community/Qwen3.5-9B-MLX-4bit -- claude --resume
```

<div align="justify">
  <p>
    Claude Code's default tools add about 30k tokens to every request, which a small local model has to read before each reply. Restricting them with <code>--strict-mcp-config --tools Bash,Read,Edit,Write,Glob,Grep</code> after <code>--</code> brings that down to about 5k and makes every turn noticeably faster.
  </p>
</div>

<br/>

<h2>How it works</h2>

<div align="justify">
  <p>
    <b>Claude Code speaks Anthropic, <code>mlx_lm.server</code> speaks OpenAI.</b> The proxy translates between the two as responses stream in &mdash; text, thinking and tool calls alike &mdash; and smooths over the few spots where Claude Code's requests would otherwise trip up a local model. It keeps the system prompt stable so the server's prompt cache can skip work it has already done, which is what keeps every turn after the first one quick.
  </p>

  <p>
    <b>It has just two flags of its own.</b> <code>--context</code> caps the context window, which otherwise comes from the model's config, and <code>--listen</code> sets the port. Run without <code>--</code>, it stays up on <code>127.0.0.1:41817</code> for any tool that accepts an <code>ANTHROPIC_BASE_URL</code>. If something goes wrong, the server and proxy logs are waiting in <code>~/.cache/claude-code-mlx-proxy.log</code>.
  </p>
</div>
