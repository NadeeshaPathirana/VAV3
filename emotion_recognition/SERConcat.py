import torch
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from collections import deque, Counter
import time

# this SER concatenate emotions over multiple utterances using confidence for a given emotion
class SERConcat:
    def __init__(self):
        self.model_path = (
            "C:/Users/220425722/Desktop/Python/Emotion Recognition/Repeat_Models/S3prl/Model_2.1/"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HubertForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        ).to(self.device)

        self.model.eval()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        self.emotion_labels = {
            0: "Anger",
            1: "Happiness",
            2: "Sadness",
            3: "Neutral"
        }

        self.max_length = 32000
        self.temperature = 1.0
        self.min_confidence = 0.30

        self.history_size = 4
        self.emotion_history = deque(maxlen=self.history_size)
        self.current_emotion = "Neutral"

    def _smooth_emotion(self):
        if not self.emotion_history:
            return self.current_emotion

        scores = {}
        for emotion, confidence in self.emotion_history:
            scores[emotion] = scores.get(emotion, 0.0) + confidence

        best_emotion = max(scores, key=scores.get)
        return best_emotion

    def predict_emotion(self, file_path):
        start_time = time.time()

        speech, _ = librosa.load(file_path, sr=16000)

        if len(speech) == 0:
            return self.current_emotion

        speech = librosa.util.normalize(speech)

        inputs = self.feature_extractor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits / self.temperature, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=-1)

            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

        if confidence < self.min_confidence:
            raw_emotion = "Neutral"
        else:
            raw_emotion = self.emotion_labels[predicted_idx]

        self.emotion_history.append((raw_emotion, confidence))
        self.current_emotion = self._smooth_emotion()

        elapsed = time.time() - start_time
        print(f"Predicted: {raw_emotion} ({confidence:.3f})")
        print(f"Smoothed: {self.current_emotion}")
        print(f"Processing time: {elapsed:.3f}s")

        return self.current_emotion