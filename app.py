import time
import threading
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from scipy.io import wavfile
import webview
import re
import random

from emotion_recognition.SpeechEmotionRecognizer import SpeechEmotionRecognizer

from tts import speechify_voice_service as vs
# from tts import coqui_voice_service as vs
from rag.AIVA_Chroma_2 import AIVA_Chroma_2
ai_assistant = AIVA_Chroma_2()
recognizer = SpeechEmotionRecognizer()
# recognizer = SpeechEmotionRecognizerV2()

main_window = None

assistant_state = {
    "running": False,
    "paused": False,
    "status": "Cai is Idle"
}

start_event = threading.Event()

shutdown_event = threading.Event()

DEFAULT_MODEL_SIZE = "small"  # set from medium to small to improve speed of the transcription
DEFAULT_CHUNK_LENGTH = 0.5  # smaller this value -> audio recording is efficient, but can only record very small chunks

def update_ui_status(text):
    assistant_state["status"] = text
    try:
        if main_window is not None:
            main_window.evaluate_js(f"updateStatus('{text}')")
    except Exception as e:
        print("JS update error:", e)

def is_silence(data, threshold=500): # threshold = max amplitude
    """Check if audio data contains silence."""
    return np.max(np.abs(data)) < threshold
def record_audio_chunk(stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    start_time = time.time()  # Start time measurement
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Check for silence
    if is_silence(audio_data):
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        # print(f"Record Audio Execution Time - is silence: {execution_time:.2f} seconds")  # Print the total execution time
        return None  # Indicate silence
    else:
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        # print(f"Record Audio Execution Time - with audio: {execution_time:.2f} seconds")  # Print the total execution time
        return audio_data  # Return audio chunk

def detect_pause(start_time, pause_duration=0.5): # this will be checked after a silence to see if the silence continues. then the total pause time = 0.5 + DEFAULT_CHUNK_LENGTH
    """Check if silence has lasted long enough to trigger a stop."""
    return (time.time() - start_time) >= pause_duration

def is_valid_response(text):
    """Check if response is valid conversational text."""
    text = text.strip()

    # Math/equation patterns
    invalid_patterns = [
        r'\*\*\d+', r'[a-z]\([a-z]\)\s*=', r'=\s*-?\d+\*',
        r'Give\s+[a-z]\(\d+\)', r'Solve\s+-?\d+', r'Question:\s*[A-Z]',
        r'Q:\s*[A-Z]', r'\d+\s*[\+\-\*/]\s*\d+\s*=', r'Let\s+[a-z]\([a-z]\)'
    ]

    for pattern in invalid_patterns:
        if re.search(pattern, text):
            return False

    if len(text) < 10 or len(text.split()) < 3:
        return False

    return True


def clean_response(text, context="general"):
    """Filter and clean response, provide fallback if needed."""
    if is_valid_response(text):
        return text

    print(f"\n⚠️ FILTERED INVALID RESPONSE: {text[:100]}...")

    if context == "intro":
        return "Hello! I'm Cai, and I'm here to chat with you. How are you feeling today?"

    # Gentle, warm fallbacks for elderly users
    fallbacks = [
        "I'm so sorry, dear. I didn't catch that. Could you tell me again?",
        "Pardon me, I missed what you said. Would you mind repeating it?",

        "Oh, I'm sorry - I didn't hear you clearly. What did you say?",
        "Forgive me, I didn't quite catch that. Could you say it once more?",
        "I apologize, I lost track of what you were saying. Could you repeat that?",
        "I'm sorry, I had a moment there. What were you telling me?",
    ]

    return random.choice(fallbacks)

def assistant_loop():
    print("Assistant thread ready, waiting for Start")
    update_ui_status("Cai is Idle")

    while not shutdown_event.is_set():
        start_event.wait()  # ⬅️ waits for Start

        if shutdown_event.is_set():
            break

        print("Conversation started")

        # === INIT ONLY AFTER START ===
        model = WhisperModel("small.en", device="cuda", compute_type="float16")

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        # Intro
        update_ui_status("Cai is Speaking")
        response = ai_assistant.interact_with_llm(
            "Introduce yourself and ask the user how they are doing today. Then ask the user what they'd like to talk about.",
            "Neutral")
        print("Conversation Starter:", response)
        response = clean_response(response, context="intro")

        stream.stop_stream()
        vs.play_text_to_speech(response, "Neutral")
        stream.start_stream()

        # === MAIN CONVERSATION LOOP ===
        while assistant_state["running"] and not shutdown_event.is_set():

            if assistant_state["paused"]:
                update_ui_status("Cai is Paused")
                time.sleep(0.1)
                continue

            update_ui_status("Cai is Listening")

            audio_chunks = []
            silence_start = None

            while True:
                if assistant_state["paused"] or not assistant_state["running"]:
                    break

                chunk = record_audio_chunk(stream)

                if chunk is not None:
                    audio_chunks.append(chunk)
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif detect_pause(silence_start):
                        break

            if not audio_chunks:
                continue

            full_audio = np.concatenate(audio_chunks)
            wavfile.write("full_audio.wav", 16000, full_audio)

            update_ui_status("Cai is Thinking")

            emotion = recognizer.predict_emotion("full_audio.wav")
            print(f"Predicted Emotion: {emotion}")

            segments, _ = model.transcribe("full_audio.wav", beam_size=3)
            transcription = " ".join(s.text for s in segments)
            print("User: ", transcription)

            if not transcription.strip():
                continue

            update_ui_status("Cai is Speaking")
            response = ai_assistant.interact_with_llm(transcription, emotion)
            response = clean_response(response, context="conversation")

            print("CAI: ", response)
            stream.stop_stream()
            vs.play_text_to_speech(response, "Neutral")
            stream.start_stream()

        # === CLEANUP SESSION ===
        stream.stop_stream()
        stream.close()
        audio.terminate()

        start_event.clear()
        update_ui_status("Cai is Idle")

    print("Assistant thread exiting")

class UIApi:

    def start(self):
        if not assistant_state["running"]:
            assistant_state["running"] = True
            assistant_state["paused"] = False
            start_event.set()
            update_ui_status("Cai is Starting")

    def pause(self):
        assistant_state["paused"] = not assistant_state["paused"]
        update_ui_status("Cai is Paused" if assistant_state["paused"] else "Cai is Listening")

    def stop(self):
        print("UI → Stop")

        shutdown_event.set()
        assistant_state["running"] = False
        assistant_state["paused"] = False

        update_ui_status("Cai is Stopping")

        # Close the window safely
        if main_window is not None:
            main_window.destroy()


if __name__ == "__main__":
    threading.Thread(target=assistant_loop, daemon=True).start()

    main_window = webview.create_window(
        "Voice Assistant User Study",
        "ui.html",
        js_api=UIApi(),
        width=480,
        height=360
    )

    webview.start()

