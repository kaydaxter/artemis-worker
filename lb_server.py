"""Modo LOAD BALANCER del worker de Artemis (vLLM directo, sin cola).

RunPod LB enruta HTTP directo al worker: el gateway hace UNA conexion SSE
passthrough a /v1/chat/completions (1 subrequest en Cloudflare Free, vs ~46
del polling de la cola). Este lanzador:

1. Levanta un shim de health en PORT_HEALTH: /ping -> 204 mientras vLLM
   carga (initializing), 200 cuando /health de vLLM responde (healthy).
2. Lanza el api_server de vLLM en PORT (la auth la pone el edge de RunPod
   con la API key de la cuenta; vLLM va sin --api-key porque el worker no
   es alcanzable directamente).

Se activa via dockerArgs del template LB: `python3 -u /lb_server.py`.
El modo cola (handler.py) sigue siendo el CMD por defecto de la imagen.
"""
import http.server
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

print("[lb] lb_server.py arrancando", flush=True)

MODEL = os.environ.get("MODEL_NAME", "TigerKay/Artemis-31B-v1i-fp8")
PORT = int(os.environ.get("PORT", "8000"))
PORT_HEALTH = int(os.environ.get("PORT_HEALTH", "8001"))
VLLM = f"http://127.0.0.1:{PORT}"


def vllm_ready():
    try:
        with urllib.request.urlopen(f"{VLLM}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


class Ping(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/ping":
            self.send_response(404)
            self.end_headers()
            return
        # 200 = healthy (enruta trafico) · 204 = initializing (espera)
        self.send_response(200 if vllm_ready() else 204)
        self.end_headers()

    def log_message(self, *a):
        pass  # higiene: nada de ruido por health checks


def health_thread():
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT_HEALTH), Ping)
    srv.serve_forever()


threading.Thread(target=health_thread, daemon=True).start()
print(f"[lb] health shim en :{PORT_HEALTH}/ping", flush=True)

cmd = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--served-model-name", "artemis",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--dtype", os.environ.get("DTYPE", "bfloat16"),
    "--max-model-len", os.environ.get("MAX_MODEL_LEN", "65536"),
    "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"),
]
extra = os.environ.get("VLLM_EXTRA_ARGS", "").strip()
if extra:
    cmd += shlex.split(extra)

print(f"[lb] lanzando vLLM en :{PORT} (modelo: {MODEL})", flush=True)
try:
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
except Exception:
    traceback.print_exc()
    time.sleep(60)
    sys.exit(1)

rc = proc.wait()
print(f"[lb] vLLM termino con exit {rc}", flush=True)
time.sleep(60)  # que el log se capture antes de que RunPod recicle
sys.exit(rc or 1)
