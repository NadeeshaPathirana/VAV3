from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa

# Load model and feature extractor
model = HubertForSequenceClassification.from_pretrained(
    "NadeeshaP/older-adult-speech-emotion-hubert"
)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "NadeeshaP/older-adult-speech-emotion-hubert"
)

model.eval()

# Load and preprocess audio (must be 16kHz)
audio, sr = librosa.load("full_audio.wav", sr=16000, mono=True)

# Extract features
inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

# Run inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get predicted emotion
predicted_id = torch.argmax(logits, dim=-1).item()
labels = ["Anger", "Happiness", "Sadness", "Neutral"]
print(f"Predicted emotion: {labels[predicted_id]}")