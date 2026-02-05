from transformers import *
import librosa
import torch


# the code provided in Hugging Face is not working

class SpeechEmotionRecognizerV2:

    def __init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "r-f/wav2vec-english-speech-emotion-recognition")
        self.model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

    def predict_emotion(self, audio_path):
        audio, rate = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(inputs.input_values)
            predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1),
                                                      dim=-1)  # Average over sequence length
            predicted_label = torch.argmax(predictions, dim=-1)
            print(f"predicted_label:{predicted_label}")
            emotion = self.model.config.id2label[predicted_label.item()]
        return emotion
