import time
from collections import deque

import librosa
import numpy as np
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# this SER concatenate emotions over multiple utterances using a sliding window and probabilities for a given emotion + weights
class SERConcat_2:
    def __init__(
        self,
        sample_rate=16000,
        max_length=32000,
        temperature=1.0,
        min_confidence=0.30,
        window_size=3,
        use_normalization=True,
    ):
        self.model_path = ("C:/Users/220425722/Desktop/Python/Emotion Recognition/Repeat_Models/S3prl/Model_2.1/")
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.window_size = window_size
        self.use_normalization = use_normalization

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HubertForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        ).to(self.device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        self.model.eval()

        self.emotion_labels = {
            0: "Anger",
            1: "Happiness",
            2: "Sadness",
            3: "Neutral"
        }

        self.num_labels = len(self.emotion_labels)
        self.prob_buffer = deque(maxlen=self.window_size)
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0

        print(f"Model loaded: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Labels: {self.emotion_labels}")
        print(f"Feature extractor sampling rate: {getattr(self.feature_extractor, 'sampling_rate', 'unknown')}")

    def _load_audio(self, file_path):
        speech, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)

        if speech is None or len(speech) == 0:
            return None

        if self.use_normalization:
            speech = librosa.util.normalize(speech)

        return speech

    def _aggregate_probs(self):
        if len(self.prob_buffer) == 0:
            return self.current_emotion, self.current_confidence

        buffer_list = list(self.prob_buffer)
        n = len(buffer_list)

        # More recent utterances get higher weight
        # e.g. for 4 utterances: weights = [1, 2, 3, 4]
        weights = np.arange(1, n + 1, dtype=float)
        weights /= weights.sum()  # normalise to sum to 1

        # Weighted average instead of simple average
        averaged = np.sum(
            [w * p for w, p in zip(weights, buffer_list)], axis=0
        )

        best_idx = int(np.argmax(averaged))
        best_confidence = float(averaged[best_idx])

        if best_confidence < self.min_confidence:
            return "Neutral", best_confidence

        return self.emotion_labels[best_idx], best_confidence

    def predict_emotion(self, file_path, return_details=False):
        start_time = time.time()

        speech = self._load_audio(file_path)
        if speech is None:
            if return_details:
                return {
                    "emotion": self.current_emotion,
                    "confidence": self.current_confidence,
                    "raw_label": self.current_emotion,
                    "raw_confidence": self.current_confidence,
                    "buffer_len": len(self.prob_buffer),
                    "elapsed_sec": 0.0,
                }
            return self.current_emotion

        inputs = self.feature_extractor(
            speech,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits / self.temperature, dim=-1)

        probs_np = probs.squeeze(0).detach().cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))
        raw_confidence = float(probs_np[predicted_idx])
        raw_label = self.emotion_labels[predicted_idx]

        self.prob_buffer.append(probs_np)

        self.current_emotion, self.current_confidence = self._aggregate_probs()

        elapsed = time.time() - start_time

        print(f"Raw logits: {logits.detach().cpu().numpy()}")
        print(f"Raw prediction: {raw_label} ({raw_confidence:.3f})")
        print(f"Buffer size: {len(self.prob_buffer)}/{self.window_size}")
        print(f"Smoothed emotion: {self.current_emotion} ({self.current_confidence:.3f})")
        print(f"Processing time: {elapsed:.3f}s")

        if return_details:
            return {
                "emotion": self.current_emotion,
                "confidence": self.current_confidence,
                "raw_label": raw_label,
                "raw_confidence": raw_confidence,
                "probs": probs_np.tolist(),
                "buffer_len": len(self.prob_buffer),
                "elapsed_sec": elapsed,
            }

        return self.current_emotion

    def reset(self):
        self.prob_buffer.clear()
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0