from transformers import *
import librosa
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")


def predict_emotion(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(inputs.input_values)
        predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)  # Average over sequence length
        predicted_label = torch.argmax(predictions, dim=-1)
        emotion = model.config.id2label[predicted_label.item()]
    return emotion


emotion = predict_emotion("C:/Users/220425722/Desktop/Python/Emotion Recognition/ravdess-emotional-speech-audio/versions/1/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav")
print(f"Predicted emotion: {emotion}")