# ccceck

This repository provides command line utilities built on top of [WhisperX](https://github.com/m-bain/whisperX) for speech-to-text transcription.

* `transcriber_main.py` – batch transcription script with optional alignment and diarization.
* `transcriber_start.py` – FastAPI service exposing HTTP endpoints for transcription.
* `stereo_dialog_start.py` – FastAPI service for stereo calls that returns the transcript as plain text.
* `stereo_dialog_main.py` – batch transcription script for stereo calls that reuses loaded models.

Example usage for a stereo file:

```bash
python stereo_dialog_main.py call.wav
```

The output prints time stamped segments in start order with speaker labels.

## Running with Docker

You can containerize the FastAPI service to make the environment reproducible. The
provided `Dockerfile` installs the Python dependencies, WhisperX, and system
packages such as `ffmpeg` that are required for audio processing.

1. Build the image:

   ```bash
   docker build -t ccceck .
   ```

2. Start the service, exposing port `8000` locally:

   ```bash
   docker run --rm -p 8000:8000 ccceck
   ```

   If you have a CUDA-capable GPU available and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
   installed, you can add `--gpus all` to the `docker run` command to allow
   WhisperX to use the GPU inside the container.

3. Once the container is running, the API is available at `http://localhost:8000`.
   You can hit `GET /health` to verify the model/device configuration and use the
   `/stereo` or `/mono` endpoints for transcription requests.

To persist inputs/outputs or pre-downloaded models between container runs, mount
host directories into the container, for example:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  ccceck
```

This shares the repository's `data` directory with the container so generated
transcripts are available on the host system.
