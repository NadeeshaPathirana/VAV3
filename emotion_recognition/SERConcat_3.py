import time
from collections import deque

import librosa
import numpy as np
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# this SER concatenate emotions over multiple utterances using rolling window + confidence + probability + weights

class SERConcat_3:
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

    def reset(self):
        self.prob_buffer.clear()
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0

    def _load_audio(self, file_path):
        speech, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        if speech is None or len(speech) == 0:
            return None
        if self.use_normalization:
            speech = librosa.util.normalize(speech)
        return speech

    def _smooth_probs(self):
        if len(self.prob_buffer) == 0:
            return None

        buffer_list = list(self.prob_buffer)  # oldest first, newest last

        # Assign weights: 1, 2, 3, ... so the last (most recent) has the highest weight
        weights = np.arange(1.0, len(buffer_list) + 1)

        # Weighted average: sum(weight * prob_vector) / sum(weights)
        weighted_sum = np.sum([w * p for w, p in zip(weights, buffer_list)], axis=0)
        smoothed = weighted_sum / np.sum(weights)

        smoothed = smoothed / (np.sum(smoothed) + 1e-12)  # normalize
        return smoothed

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
                    "smoothed_probs": None,
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
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits / self.temperature, dim=-1)

        probs_np = probs.squeeze(0).detach().cpu().numpy()
        raw_idx = int(np.argmax(probs_np))
        raw_label = self.emotion_labels[raw_idx]
        raw_confidence = float(probs_np[raw_idx])

        self.prob_buffer.append(probs_np)

        smoothed_probs = self._smooth_probs()
        if smoothed_probs is None:
            self.current_emotion = raw_label
            self.current_confidence = raw_confidence
        else:
            best_idx = int(np.argmax(smoothed_probs))
            best_confidence = float(smoothed_probs[best_idx])

            if best_confidence < self.min_confidence:
                self.current_emotion = "Neutral"
                self.current_confidence = best_confidence
            else:
                self.current_emotion = self.emotion_labels[best_idx]
                self.current_confidence = best_confidence

        elapsed = time.time() - start_time

        print(f"Raw prediction: {raw_label} ({raw_confidence:.3f})")
        print(f"Smoothed emotion: {self.current_emotion} ({self.current_confidence:.3f})")
        print(f"Buffer size: {len(self.prob_buffer)}/{self.window_size}")
        print(f"Processing time: {elapsed:.3f}s")

        if return_details:
            return {
                "emotion": self.current_emotion,
                "confidence": self.current_confidence,
                "raw_label": raw_label,
                "raw_confidence": raw_confidence,
                "smoothed_probs": smoothed_probs.tolist() if smoothed_probs is not None else None,
                "raw_probs": probs_np.tolist(),
                "buffer_len": len(self.prob_buffer),
                "elapsed_sec": elapsed,
            }

        return self.current_emotion