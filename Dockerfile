# Artemis-31B-v1i (Gemma 4 dense) served via llama.cpp for RunPod Serverless.
# Quant: Q8_0 GGUF from BeaverAI/Artemis-31B-v1i-GGUF (~30.4 GB → needs a 48GB GPU).
#
# Why llama.cpp and not vLLM: the model only ships as GGUF, and llama.cpp is the
# stack where the community validates Artemis. It also sidesteps the Gemma-4
# fp8-dynamic gibberish bug in vLLM (#39049).
#
# Build (via your GitHub Actions → GHCR), then point a RunPod Serverless
# endpoint at the resulting image and set the secrets/vars below.
#
# NOTE: pin the base image to a digest once it builds clean. Tag may drift.
FROM ghcr.io/ggml-org/llama.cpp:full-cuda

# The base image keeps binaries AND their shared libs in /app (WORKDIR); its
# ENTRYPOINT is a dispatcher (/app/tools.sh) that sets these up. We reset the
# entrypoint and call llama-server ourselves, so /app must be on PATH (find the
# binary) AND on LD_LIBRARY_PATH (find libllama-server-impl.so etc.) — without
# the latter, llama-server dies with exit 127 "cannot open shared object file".
ENV PATH="/app:${PATH}"
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

# Python + RunPod SDK + HF downloader on top of the llama.cpp image.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages runpod requests huggingface_hub hf_transfer

# Model + serving config (override in the RunPod endpoint if needed).
ENV MODEL_REPO=BeaverAI/Artemis-31B-v1i-GGUF \
    MODEL_FILE=Artemis-31B-v1i-Q8_0.gguf \
    CTX_SIZE=32768 \
    GPU_LAYERS=999 \
    PARALLEL=4 \
    LLAMA_PORT=8080

# Gemma 4 defaults to *thinking* mode: the reply lands in reasoning_content and
# `text` comes back empty. Bake the off-switch into the image so a redeploy
# can't forget it (the RunPod template sets these too, but that step is manual).
ENV LLAMA_ARG_THINK_BUDGET=0 \
    LLAMA_ARG_REASONING=off

# HF_TOKEN is set as a RunPod *secret* (needed only if the repo is private).
# The GGUF is downloaded on first boot into a volume/local dir (see README:
# bake it into the image later if cold-start download hurts).

COPY handler.py /handler.py

# The llama.cpp base image sets its own ENTRYPOINT; reset it so our handler runs.
ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
