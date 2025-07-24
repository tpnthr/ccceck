# ccceck

This repository provides command line utilities built on top of [WhisperX](https://github.com/m-bain/whisperX) for speech-to-text transcription.

* `transcriber_main.py` – batch transcription script with optional alignment and diarization.
* `transcriber_start.py` – FastAPI service exposing HTTP endpoints for transcription.
* `stereo_dialog.py` – helper that takes a stereo call recording, splits the two channels and produces a single dialogue transcript with the left channel labeled as **caller** and the right channel labeled as **client**.

Example usage for a stereo file:

```bash
python stereo_dialog.py call.wav
```

The output prints time stamped segments in start order with speaker labels.
