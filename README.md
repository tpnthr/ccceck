# ccceck

Utilities for building GPU-accelerated speech-to-text services on top of
[WhisperX](https://github.com/m-bain/whisperX). The repository contains CLI
scripts and FastAPI services for mono and stereo audio transcription.

This document provides end-to-end instructions for deploying the FastAPI
service on **Ubuntu 24.04** with an **NVIDIA RTX 5090** GPU inside Docker while
keeping driver, CUDA, and PyTorch versions compatible. The steps cover both host
configuration and day-to-day Docker usage so you can reproduce the environment
consistently across machines.

## 1. Prerequisites

- Ubuntu 24.04 LTS with administrative (sudo) privileges.
- An NVIDIA RTX 5090 connected directly to the host.
- Internet access to reach the NVIDIA, Docker, and Python package repositories.
- At least 40 GB of free disk space for Docker images, CUDA libraries, and model
  weights.

The provided `Dockerfile` is based on
`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, which requires an R550+ NVIDIA
driver on the host (CUDA 12.4 compatibility). The RTX 5090 is expected to ship
with official support in the R555 branch. Make sure your host driver is **≥
555.xx** so that the container can access the GPU. If you need to run an older
driver, swap the base image and CUDA-enabled wheels to a matching version (see
[Section 10](#10-updating-drivers-or-cuda)).

## 2. Update the system and install the NVIDIA driver

```bash
sudo apt update && sudo apt upgrade -y
sudo ubuntu-drivers install nvidia:555
sudo reboot
```

> **Why 555?** CUDA 12.4 containers require at least the R550 driver series.
> Using the latest available 555 build on Ubuntu 24.04 ensures the RTX 5090 is
> fully recognized and that CUDA forward-compatibility for future minor versions
> (12.x) remains intact.

After the reboot verify the installation:

```bash
nvidia-smi
```

The output should list the RTX 5090 and show a driver version ≥ 555.xx. If the
command fails, resolve the driver issue before continuing.

> 💡 **Air-gapped or offline installs**: download the driver `.run` installer on
> another machine, transfer it to the server, and install with `sudo sh
> NVIDIA-Linux-x86_64-555.xx.run --silent`. Be sure to reinstall the driver
> after every kernel upgrade.

## 3. Install Docker Engine

```bash
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

The `newgrp docker` command reloads your shell group membership so Docker can be
run without `sudo`.

## 4. Install the NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Confirm Docker sees the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The command should print the RTX 5090 details from inside the container.

## 5. Clone the repository

```bash
git clone https://github.com/your-org/ccceck.git
cd ccceck
```

> If you are developing locally, keep the repository inside a project directory
> (for example `~/projects/ccceck`) so the relative bind mounts in later steps
> line up with your host filesystem layout.

## 6. Build the GPU-enabled Docker image

The Dockerfile installs CUDA-enabled PyTorch 2.2.2 wheels (`cu121`) compatible
with CUDA 12.4 runtime libraries. Build the image with a descriptive tag:

```bash
docker build -t ccceck:cuda12 .
```

If you need to target a different CUDA/PyTorch combination, edit the
`Dockerfile` to use the matching base image and wheel index (for example,
`https://download.pytorch.org/whl/cu124` once PyTorch publishes CUDA 12.4
wheels). Always ensure the host driver meets the minimum requirement of the CUDA
runtime you select. When in doubt, consult the [CUDA compatibility
matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) and ensure the
driver version on the host is greater than or equal to the minimum listed for
your CUDA runtime.

## 7. Run the FastAPI service with GPU access

```bash
docker run --rm \ 
  --gpus all \ 
  -p 8000:8000 \ 
  -v "$(pwd)/data:/app/data" \ 
  -e TRANSCRIBE_DEVICE=cuda \ 
  ccceck:cuda12
```

- `--gpus all` exposes the RTX 5090 inside the container.
- The bind mount makes local input/output files available in the container.
- `TRANSCRIBE_DEVICE=cuda` (optional) instructs WhisperX to prefer the GPU. The
  application falls back to CUDA automatically if available, but the environment
  variable makes the intent explicit.

The service listens on `http://localhost:8000`. Use the `/health` endpoint to
confirm that the GPU is detected and models are loaded on CUDA.

### Run ad-hoc commands inside the container

For administrative or debugging tasks, start an interactive shell with GPU
access:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  ccceck:cuda12 /bin/bash
```

Within the shell you can run `pytest`, invoke CLI utilities, or inspect CUDA
availability:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Remember to exit the shell when finished to release GPU resources.

## 8. Example API usage

Health check:

```bash
curl http://localhost:8000/health
```

Stereo transcription (expects `call.wav` inside the `data/` directory):

```bash
curl -X POST "http://localhost:8000/stereo" \
  -F "file=@data/call.wav"
```

## 9. Model caching and large downloads

WhisperX downloads ASR and alignment models the first time they are needed. To
avoid re-downloading, mount a persistent cache directory:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$HOME/.cache/whisperx:/root/.cache/whisperx" \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ccceck:cuda12
```

The additional mount keeps Hugging Face transformer weights persistent between
container runs, saving time when upgrading or scaling out.

## 10. Updating drivers or CUDA

When NVIDIA releases newer drivers or CUDA runtimes for the RTX 5090:

1. Upgrade the host driver first (`sudo ubuntu-drivers install nvidia:<new>`).
2. Rebuild the Docker image using a CUDA base image that matches the new driver
   compatibility matrix.
3. Update the PyTorch wheel index to the matching CUDA version if needed.
4. Re-run `docker run --rm --gpus all nvidia/cuda:<version>-base-ubuntu22.04
   nvidia-smi` to confirm the host/container handshake still works.

Keeping the host driver **ahead** of the container CUDA runtime prevents
incompatibilities such as `CUDA_ERROR_INVALID_DEVICE`.

## 11. Troubleshooting tips

- **`nvidia-smi` works on the host but not in the container** – Re-run
  `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker.
- **`CUDA driver version is insufficient` error** – Update the host driver to at
  least the minimum version required by the CUDA runtime in your Docker image.
- **Slow downloads or timeouts** – Pre-download WhisperX models on the host and
  mount the cache directory into the container as shown above.
- **`torch.cuda.is_available()` is `False` inside Docker** – Check that the
  container was started with `--gpus all`, your user belongs to the `docker`
  group, and the NVIDIA persistence daemon is running on the host.
- **Need to run batch jobs instead of the API** – Use the interactive shell
  command above and call the CLI utilities directly (for example `python
  stereo_dialog_startxx.py --input data/call.wav`).

With these steps, the ccceck FastAPI service will run on Ubuntu 24.04, leverage
an RTX 5090 via Docker, and remain compatible with NVIDIA's driver and CUDA
release cadence.
