import time

import librosa
import numpy as np
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor


class SERConcat_4:
    """
    Speech Emotion Recognition with Exponential Weighted Moving Average (EWMA)
    over softmax probability vectors, following:

        EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}

    where:
      - x_t: current probability vector (softmax over emotions for this utterance)
      - EMA_{t-1}: previous smoothed probability vector
      - alpha: weight on CURRENT probabilities (0 < alpha <= 1)
    """

    def __init__(
        self,
        sample_rate=16000,
        max_length=32000,
        temperature=1.0,
        min_confidence=0.30,
        ema_alpha=0.8,        # weight on current emotion
        use_normalization=True,
    ):
        """
        ema_alpha (0 < ema_alpha <= 1):

        - Close to 1.0  -> follow current utterance strongly, little smoothing.
        - Around 0.5    -> balance between current and history.
        - Close to 0.0  -> very strong memory, slow to change (not recommended
                           for a reactive assistant).
        A practical range for a voice assistant is 0.5–0.8; you can tune this.
        """
        self.model_path = ("C:/Users/220425722/Desktop/Python/Emotion Recognition/Repeat_Models/S3prl/Model_2.1/")
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.ema_alpha = ema_alpha
        self.use_normalization = use_normalization

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and feature extractor
        self.model = HubertForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        ).to(self.device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        self.model.eval()

        # Emotion labels
        self.emotion_labels = {
            0: "Anger",
            1: "Happiness",
            2: "Sadness",
            3: "Neutral",
        }

        self.num_labels = len(self.emotion_labels)

        # EMA state: None until we see the first utterance
        self.smoothed_probs = None

        # Public state for the assistant
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0

        print(f"Model loaded: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Labels: {self.emotion_labels}")
        print(f"EMA alpha (weight on current): {self.ema_alpha}")

    def reset(self):
        """
        Reset EMA and current state, e.g., at the start of a new conversation.
        """
        self.smoothed_probs = None
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0

    def _load_audio(self, file_path):
        """
        Load and optionally normalize audio from file_path.
        """
        speech, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        if speech is None or len(speech) == 0:
            return None
        if self.use_normalization:
            speech = librosa.util.normalize(speech)
        return speech

    def _update_ema(self, current_probs):
        """
        Update EMA (exponential moving average) over probability vector following the Medium article:

            EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}

        where:
          - x_t          = current_probs
          - EMA_{t-1}    = self.smoothed_probs (previous EMA)
          - EMA_t        = new self.smoothed_probs
        """
        if self.smoothed_probs is None:
            # Initialize EMA with the first observation (like Day 1 = 100)
            self.smoothed_probs = current_probs.copy()
        else:
            alpha = self.ema_alpha
            self.smoothed_probs = (
                alpha * current_probs + (1.0 - alpha) * self.smoothed_probs
            )

        # Normalize in case of small numerical drift
        s = np.sum(self.smoothed_probs)
        if s > 0:
            self.smoothed_probs = self.smoothed_probs / s

    def predict_emotion(self, file_path, return_details=False):
        """
        Run SER on one utterance (audio file), update EMA, and return
        the smoothed emotion and confidence.
        """
        start_time = time.time()

        speech = self._load_audio(file_path)
        if speech is None:
            if return_details:
                return {
                    "emotion": self.current_emotion,
                    "confidence": self.current_confidence,
                    "raw_label": self.current_emotion,
                    "raw_confidence": self.current_confidence,
                    "smoothed_probs": None if self.smoothed_probs is None else self.smoothed_probs.tolist(),
                    "elapsed_sec": 0.0,
                }
            return self.current_emotion

        # Feature extraction
        inputs = self.feature_extractor(
            speech,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Model inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits / self.temperature, dim=-1)

        probs_np = probs.squeeze(0).detach().cpu().numpy()

        # Raw per-utterance prediction (no smoothing)
        raw_idx = int(np.argmax(probs_np))
        raw_label = self.emotion_labels[raw_idx]
        raw_confidence = float(probs_np[raw_idx])

        # Update EMA with current probabilities
        self._update_ema(probs_np)

        # Decide from EMA-smoothed probabilities
        if self.smoothed_probs is None:
            best_idx = raw_idx
            best_confidence = raw_confidence
        else:
            best_idx = int(np.argmax(self.smoothed_probs))
            best_confidence = float(self.smoothed_probs[best_idx])

        # Confidence threshold on smoothed estimate
        if best_confidence < self.min_confidence:
            self.current_emotion = "Neutral"
            self.current_confidence = best_confidence
        else:
            self.current_emotion = self.emotion_labels[best_idx]
            self.current_confidence = best_confidence

        elapsed = time.time() - start_time

        print(f"Raw prediction: {raw_label} ({raw_confidence:.3f})")
        print(f"Smoothed emotion: {self.current_emotion} ({self.current_confidence:.3f})")
        print(f"EMA alpha (current weight): {self.ema_alpha}")
        print(f"Processing time: {elapsed:.3f}s")

        if return_details:
            return {
                "emotion": self.current_emotion,
                "confidence": self.current_confidence,
                "raw_label": raw_label,
                "raw_confidence": raw_confidence,
                "smoothed_probs": self.smoothed_probs.tolist() if self.smoothed_probs is not None else None,
                "raw_probs": probs_np.tolist(),
                "elapsed_sec": elapsed,
            }

        return self.current_emotion