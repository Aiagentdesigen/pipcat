import sounddevice as sd
import numpy as np 
from faster_whisper import WhisperModel 


SR = 16000 
DUR = 6

print("recording...")

audio = sd.rec(int(SR * DUR), samplerate=SR, channels=1, dtype="int16")
sd.wait()

audio_f32 = audio.reshape(-1).astype(np.float32) / 32768.0 
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_f32, language="en")

text = "".join(s.text for s in segments).strip()
print(text)
