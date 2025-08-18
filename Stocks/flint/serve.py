# serve.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path
import subprocess
import sys
import shlex
import os
import re

app = FastAPI()
ROOT = Path(__file__).resolve().parent

# Strip ANSI so the browser doesn't show escape codes. Toggle to keep color if you add ansi2html client-side.
ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Project Flint – Web CLI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --bg:#0b0f14; --panel:#0e1520; --accent:#5aa9ff; --text:#e6edf3; --muted:#8b949e; --border:#1c2633; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
           background:var(--bg); color:var(--text); }
    .wrap { max-width: 960px; margin: 28px auto; padding: 0 16px; }
    h1 { font-size: 18px; margin: 0 0 14px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
    .form-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 12px;
      align-items: end;
    }
    .col-span-3 { grid-column: 1 / -1; }
    label { font-size:12px; color:var(--muted); margin-bottom:6px; display:block; }
    input[type=text], select {
      width: 100%; background:#0b1320; color:var(--text); border:1px solid #1a2230;
      border-radius:8px; padding:9px 10px; outline:none;
    }
    input[type=text]:focus, select:focus { border-color: var(--accent); }
    .flags { display:flex; gap:16px; align-items:center; flex-wrap: wrap; }
    .flags label { display:flex; align-items:center; gap:6px; cursor:pointer; font-size: 13px; color: var(--text); }
    .btnrow { display:flex; gap:10px; margin-top:10px; flex-wrap: wrap; }
    button {
      background: var(--accent); color:#001833; border:0; padding:10px 14px; border-radius:10px; cursor:pointer; font-weight:700;
    }
    button.secondary { background: #1d2a3a; color: var(--text); border:1px solid #26364c; }
    button:disabled { opacity:.6; cursor:not-allowed; }
    .status { font-size:12px; color:var(--muted); margin-top:6px; }
    pre.terminal {
      margin-top: 16px; background:#03070b; border:1px solid #0d1520; border-radius:12px; padding:12px;
      height: 480px; overflow:auto; white-space: pre-wrap; line-height:1.35;
      overflow-wrap: anywhere; word-break: break-word; font-variant-ligatures: none;
    }
    .hint { font-size:12px; color:var(--muted); margin-top:8px; }
    @media (max-width: 900px) { .form-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Project Flint – Web CLI</h1>
    <div class="card">
      <div class="form-grid">
        <div>
          <label>Tickers (space-separated)</label>
          <input id="tickers" type="text" placeholder="e.g., AAPL MSFT TSLA" />
        </div>
        <div>
          <label>Model</label>
          <select id="model">
            <option value="ensemble">ensemble</option>
            <option value="xgb">xgb</option>
            <option value="rf">rf</option>
          </select>
        </div>
        <div>
          <label>Flags</label>
          <div class="flags">
            <label><input id="force" type="checkbox" /> --force-reprocess</label>
            <label><input id="profile" type="checkbox" /> --profile</label>
          </div>
        </div>
        <div class="col-span-3">
          <label>Raw CLI Args (optional)</label>
          <input id="raw" type="text" placeholder='Overrides fields above, e.g. "AAPL -m rf --force-reprocess"' />
        </div>
      </div>

      <div class="btnrow">
        <button id="run">Run</button>
        <button id="stop" class="secondary" disabled>Stop</button>
        <button id="clear" class="secondary">Clear</button>
      </div>
      <div class="status" id="status">Idle.</div>

      <pre id="term" class="terminal"></pre>
      <div class="hint">Tip: leave “Raw CLI Args” blank to use the fields above. Otherwise, the raw text is sent exactly like the CLI.</div>
    </div>
  </div>

  <script>
    const term = document.getElementById('term');
    const statusEl = document.getElementById('status');
    const btnRun = document.getElementById('run');
    const btnStop = document.getElementById('stop');
    const btnClear = document.getElementById('clear');
    const $ = id => document.getElementById(id);
    let es = null;

    function buildArgs() {
      const raw = $('raw').value.trim();
      if (raw) return raw;
      const tickers = $('tickers').value.trim();
      const model = $('model').value;
      const flags = [];
      if ($('force').checked) flags.push('--force-reprocess');
      if ($('profile').checked) flags.push('--profile');
      const parts = [];
      if (tickers) parts.push(tickers);
      if (model) parts.push('-m ' + model);
      if (flags.length) parts.push(flags.join(' '));
      return parts.join(' ').trim();
    }

    function setRunning(running) {
      btnRun.disabled = running;
      btnStop.disabled = !running;
      statusEl.textContent = running ? 'Running… streaming logs.' : 'Idle.';
    }

    function append(line) {
      term.textContent += line + "\\n";
      term.scrollTop = term.scrollHeight;
    }

    btnRun.addEventListener('click', () => {
      if (es) es.close();
      const args = buildArgs();
      if (!args) { append('Please supply tickers or raw args.'); return; }
      es = new EventSource('/stream?args=' + encodeURIComponent(args));
      setRunning(true);
      es.onmessage = (ev) => append(ev.data);
      es.addEventListener('done', (ev) => {
        append('--- process finished: ' + ev.data + ' ---');
        setRunning(false);
        es.close(); es = null;
      });
      es.onerror = () => {
        append('[error] connection closed.');
        setRunning(false);
        if (es) { es.close(); es = null; }
      };
    });

    btnStop.addEventListener('click', () => {
      if (es) { es.close(); es = null; append('--- stream closed by user ---'); }
      setRunning(false);
    });

    btnClear.addEventListener('click', () => { term.textContent = ''; });
  </script>
</body>
</html>"""


def _split_args(argline: str) -> list[str]:
    # Use POSIX-style splitting unless on Windows outside WSL.
    posix = not (os.name == "nt" and "WSL_DISTRO_NAME" not in os.environ)
    return shlex.split(argline, posix=posix)

def _sse_stream(cmd: list[str]):
    # -u for unbuffered, so logs stream immediately
    full_cmd = [sys.executable, "-u", "run.py", *cmd]
    proc = subprocess.Popen(
        full_cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        universal_newlines=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # Strip ANSI for clean browser output
            clean = ANSI.sub("", line.rstrip("\n"))
            yield f"data: {clean}\n\n"
    finally:
        rc = proc.poll()
        if rc is None:
            proc.terminate()
            rc = proc.wait(timeout=5)
        yield f"event: done\ndata: returncode={rc}\n\n"

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.get("/stream")
def stream(args: str):
    cmd = _split_args(args)
    return StreamingResponse(_sse_stream(cmd), media_type="text/event-stream")