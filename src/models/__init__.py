# def load_models(need_alignment: bool, language: str) -> None:
#     global ASR_MODEL, ALIGN_MODEL, ALIGN_META
#     log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={DEVICE}")
#     log(f"Loading WhisperX ASR '{WHISPER_MODEL}' (compute_type={COMPUTE_TYPE})")
#     ASR_MODEL = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
#
#     if need_alignment:
#         loaded = False
#         for candidate in ALIGN_MODEL_CANDIDATES:
#             try:
#                 log(f"Trying alignment model: {candidate}")
#                 ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(language_code=language, device=DEVICE, model_name=candidate)
#                 log(f"Loaded alignment model: {candidate}")
#                 loaded = True
#                 break
#             except Exception as e:
#                 warn(f"Alignment model failed: {candidate} -> {e}")
#         if not loaded:
#             warn("All alignment candidates failed; proceeding WITHOUT alignment.")
#             ALIGN_MODEL = None
#             ALIGN_META = None
#
#     # Warmup with second of silence
#     try:
#         import numpy as np, soundfile as sf
#         tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         sr = 16000
#         sf.write(tmp.name, np.zeros(sr, dtype="float32"), sr)
#         transcribe_stereo(tmp.name, language, bool(ALIGN_MODEL), False, False)
#         tmp.close()
#         try:
#             os.remove(tmp.name)
#         except Exception:
#             pass
#         log("Warmup complete.")
#     except Exception as e:
#         warn(f"Warmup skipped: {e}")
