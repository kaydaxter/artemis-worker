"""Worker RunPod serverless: Artemis-31B-v1i (Gemma 4) via llama.cpp / GGUF.

Mirrors the proven qwen35-worker-vllm pattern: blocking boot, flushed logs,
fatal() that sleeps 60s so the log is captured. Speaks the contract the app
expects (app/src/routes/chat.js): input {messages, sampling_params, stream} ->
non-stream {text, usage}; stream -> raw string pieces.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

print("[worker] handler.py arrancando (python OK)", flush=True)


def fatal(msg, exc=True):
    print(f"[worker][FATAL] {msg}", flush=True)
    if exc:
        traceback.print_exc()
    print("[worker] durmiendo 60s para que el log se capture...", flush=True)
    time.sleep(60)
    sys.exit(1)


try:
    import urllib.request
    import urllib.error
    import runpod
    from huggingface_hub import hf_hub_download
    print("[worker] imports OK (runpod %s)" % getattr(runpod, "__version__", "?"), flush=True)
except Exception:
    fatal("fallo importando dependencias")

MODEL_REPO = os.environ.get("MODEL_REPO", "BeaverAI/Artemis-31B-v1i-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Artemis-31B-v1i-Q8_0.gguf")
PORT = "8080"
BASE = f"http://127.0.0.1:{PORT}"
LLAMA_BIN = shutil.which("llama-server") or "/app/llama-server"

print(f"[worker] descargando {MODEL_REPO}/{MODEL_FILE} ...", flush=True)
try:
    # single copy in the HF cache (no local_dir) to avoid two 30GB copies
    MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE,
                                 token=os.environ.get("HF_TOKEN"))
    print(f"[worker] modelo en {MODEL_PATH}", flush=True)
except Exception:
    fatal("descarga del GGUF fallo")

cmd = [
    LLAMA_BIN, "-m", MODEL_PATH,
    "--host", "127.0.0.1", "--port", PORT,
    "-ngl", os.environ.get("GPU_LAYERS", "999"),
    "-c", os.environ.get("CTX_SIZE", "32768"),
    "--parallel", os.environ.get("PARALLEL", "4"),
    "--cont-batching",
]
print("[worker] lanzando llama-server: %s" % " ".join(cmd), flush=True)
t0 = time.time()
try:
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
except Exception:
    fatal("Popen de llama-server fallo")

while True:
    rc = proc.poll()
    if rc is not None:
        fatal(f"llama-server murio en el arranque con exit code {rc} "
              f"(a los {time.time()-t0:.0f}s) — su error debe estar arriba", exc=False)
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    el = int(time.time() - t0)
    if el and el % 30 < 2:
        print(f"[worker] esperando a llama-server... {el}s", flush=True)
    time.sleep(2)
print(f"[worker] llama-server LISTO en {time.time()-t0:.1f}s", flush=True)


def _req(path, body):
    return urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")


def _build_body(inp):
    sp = inp.get("sampling_params", {}) or {}
    body = {
        "messages": inp.get("messages", []),
        "max_tokens": int(sp.get("max_tokens", 1024)),
        "temperature": float(sp.get("temperature", 0.8)),
        "repeat_penalty": float(sp.get("repeat_penalty", 1.1)),
    }
    if sp.get("top_p") is not None:
        body["top_p"] = float(sp["top_p"])
    if sp.get("seed") is not None:
        body["seed"] = int(sp["seed"])
    if sp.get("stop") is not None:
        body["stop"] = sp["stop"]
    return body


def handler(job):
    """Generator: stream=True -> yield text deltas; stream=False -> one {text,usage}."""
    inp = (job or {}).get("input", {}) or {}
    body = _build_body(inp)
    try:
        if inp.get("stream"):
            body = {**body, "stream": True}
            with urllib.request.urlopen(_req("/v1/chat/completions", body), timeout=900) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {}).get("content") or ""
                    except Exception:
                        continue
                    if delta:
                        yield delta
        else:
            with urllib.request.urlopen(_req("/v1/chat/completions", body), timeout=900) as r:
                out = json.loads(r.read().decode())
            yield {"text": out["choices"][0]["message"]["content"], "usage": out.get("usage")}
    except urllib.error.HTTPError as e:
        yield {"error": f"HTTP {e.code}", "body": e.read().decode()[:800]}
    except Exception as e:
        yield {"error": f"{type(e).__name__}: {e}"}


try:
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
except Exception:
    fatal("runpod.serverless.start fallo")
