import torch
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import time


class SpeechEmotionRecognizer:
    def __init__(self):
        self.model_path = (
            "C:/Users/220425722/Desktop/Python/Emotion Recognition/Repeat_Models/S3prl/Model_2/"
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

        # VERIFY THIS MATCHES TRAINING
        self.emotion_labels = {
            0: "Anger",
            1: "Happiness",
            2: "Sadness",
            3: "Neutral"
        }

        # ~4 seconds (S3PRL-friendly)
        self.max_length = 64000

        # Inference controls
        self.temperature = 2.0
        self.min_confidence = 0.45

    def predict_emotion(self, file_path):
        start_time = time.time()

        # Load audio
        speech, _ = librosa.load(file_path, sr=16000)

        # Loudness normalization (very important for SER)
        speech = librosa.util.normalize(speech)

        # Feature extraction (NO manual padding)
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

            # Temperature scaling
            probs = torch.softmax(logits / self.temperature, dim=-1)

            confidence, predicted_idx = torch.max(probs, dim=-1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

        end_time = time.time()
        print(f"Emotion Recogniser Time: {end_time - start_time:.2f}s")
        print(f"Probs: {probs.cpu().numpy()}, confidence={confidence:.2f}")

        # Confidence-based fallback
        if confidence < self.min_confidence:
            return "Neutral"

        return self.emotion_labels[predicted_idx]
