import torch
print(torch.__version__)  # Should print 2.6.0 or higher
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-polish")
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-polish")
