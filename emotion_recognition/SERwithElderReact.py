from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torchaudio

# Load model and feature extractor
model = Wav2Vec2ForSequenceClassification.from_pretrained("./elder_react_model") # correct the locations
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./elder_react_model")
model.eval()

# Load a new audio file
speech_array, sr = torchaudio.load("example.wav")
if speech_array.shape[0] > 1:
    speech_array = torch.mean(speech_array, dim=0)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech_array = resampler(speech_array)

# Prepare input
inputs = feature_extractor(speech_array.numpy(), sampling_rate=16000, return_tensors="pt")

# Predict
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax(-1).item()

# Map to emotion label
emotion_map = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fear", 5: "disgust"}
print("Predicted emotion:", emotion_map[predicted_class_id])
