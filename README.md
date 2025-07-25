# ccceck

This repository provides command line utilities built on top of [WhisperX](https://github.com/m-bain/whisperX) for speech-to-text transcription.

* `transcriber_main.py` – batch transcription script with optional alignment and diarization.
* `transcriber_start.py` – FastAPI service exposing HTTP endpoints for transcription.
* `stereo_dialog_start.py` – FastAPI service that accepts stereo call recordings and returns a merged dialogue transcript.
* `stereo_dialog_main.py` – batch transcription script for stereo calls that reuses loaded models.

Example usage for a stereo file:

```bash
python stereo_dialog_main.py call.wav
```

The output prints time stamped segments in start order with speaker labels.
