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

# The base image keeps its binaries in /app (WORKDIR) and its ENTRYPOINT is a
# dispatcher (/app/tools.sh) — /app is NOT on PATH. We reset the entrypoint and
# call llama-server ourselves, so put /app on PATH or the handler can't find it.
ENV PATH="/app:${PATH}"

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

# HF_TOKEN is set as a RunPod *secret* (needed only if the repo is private).
# The GGUF is downloaded on first boot into a volume/local dir (see README:
# bake it into the image later if cold-start download hurts).

COPY handler.py /handler.py

# The llama.cpp base image sets its own ENTRYPOINT; reset it so our handler runs.
ENTRYPOINT []
CMD ["python3", "/handler.py"]
