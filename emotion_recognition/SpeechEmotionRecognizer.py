import torch
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import time


class SpeechEmotionRecognizer:
    def __init__(self):
        self.model_path = (
            "C:/Users/220425722/Desktop/Python/Emotion Recognition/Repeat_Models/S3prl/Model_2.1/"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HubertForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        ).to(self.device)
        print(f"Model loaded: {self.model.__class__.__name__}")
        print(f"Number of labels: {self.model.config.num_labels}")
        print(f"Model device: {next(self.model.parameters()).device}")

        # Verify the model has actual weights (not random initialization)
        sample_weight = next(self.model.parameters())
        print(f"Sample weight stats - mean: {sample_weight.mean():.4f}, std: {sample_weight.std():.4f}")

        self.model.eval()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        print(f"Feature extractor sampling rate: {self.feature_extractor.sampling_rate}")
        print(f"Feature extractor return_attention_mask: {self.feature_extractor.return_attention_mask}")

        # Add this more comprehensive check:
        print("\n=== MODEL WEIGHT ANALYSIS ===")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Check the classifier head specifically (most important for your task)
        if hasattr(self.model, 'classifier'):
            classifier_weight = self.model.classifier.weight
            print(f"Classifier weight - mean: {classifier_weight.mean():.4f}, std: {classifier_weight.std():.4f}")
            print(f"Classifier weight range: [{classifier_weight.min():.4f}, {classifier_weight.max():.4f}]")
        else:
            print("No classifier layer found - check model architecture")

        # Check if any layer has suspiciously uniform values
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'head' in name:
                print(
                    f"{name}: mean={param.mean():.4f}, std={param.std():.4f}, range=[{param.min():.4f}, {param.max():.4f}]")

        # VERIFY THIS MATCHES TRAINING
        self.emotion_labels = {
            0: "Anger",
            1: "Happiness",
            2: "Sadness",
            3: "Neutral"
        }

        # ~4 seconds (S3PRL-friendly)
        self.max_length = 32000

        # Inference controls
        self.temperature = 1.0
        # self.min_confidence = 0.45
        self.min_confidence = 0.30

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
            print(f"Raw logits: {logits.cpu().numpy()}")

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
