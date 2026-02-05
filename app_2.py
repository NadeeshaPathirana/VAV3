import os
import time
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import io
from scipy.io.wavfile import write

from tts import google_voice_service as vs
# from tts import coqui_voice_service as cs
from rag.AIVA import AIVA
from rag.AIVA_Chroma import AIVA_Chroma

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10

# ai_assistant = AIVoiceAssistant()
# ai_assistant = AIVA()
ai_assistant = AIVA_Chroma()

# V2 - trying to optimise the recording process - without reading or writing the audio file

def is_silence(data, threshold=500):
    """Check if audio data contains silence."""
    return np.max(np.abs(data)) < threshold


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    start_time = time.time()
    frames = []

    for _ in range(0, int(16000 / 4096 * chunk_length)):
        data = stream.read(4096)
        frames.append(data)

    # Convert recorded audio into a bytes object
    audio_data = b''.join(frames)

    # Convert bytes into NumPy array for silence detection
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Silence check
    if is_silence(audio_np):
        return None  # Return None to indicate silence detected

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Record Audio Execution Time: {execution_time:.2f} seconds")

    return audio_data  # Return the audio bytes instead of writing to a file


def transcribe_audio(model, audio_data):
    start_time = time.time()

    # Convert raw audio bytes into a NumPy array for processing
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Save the NumPy array as an in-memory WAV file
    wav_buffer = io.BytesIO()
    write(wav_buffer, 16000, audio_np)  # Save as WAV with 16kHz sample rate

    # Reset the buffer position before reading it for transcription
    wav_buffer.seek(0)

    # Transcribe the in-memory WAV file
    segments, info = model.transcribe(wav_buffer, beam_size=7)  # Use beam_size=1 for speed

    # Extract transcribed text
    transcription = ' '.join(segment.text for segment in segments)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Transcribe Audio Execution Time: {execution_time:.2f} seconds")

    return transcription


def main():
    model_size = DEFAULT_MODEL_SIZE + ".en" # faster_whisper model size
    # model = WhisperModel(model_size, device="cpu", compute_type="float32", num_workers=10) # cuda if using GPU, otherwise CPU
    model = WhisperModel(model_size, device="cuda", compute_type="float16")  # Use GPU if available
    # TODO: change this code to run in the server
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    user_input_transcription = ""

    try:
        while True:
            print("_")

            # Record audio in memory
            audio_data = record_audio_chunk(audio, stream)

            if audio_data is not None:
                # Transcribe the in-memory audio data
                transcription = transcribe_audio(model, audio_data)

                # Process the transcription
                print(f"User: {transcription}")
                output = ai_assistant.interact_with_llm(transcription)

                if output:
                    print(f"AI Assistant: {output}")
                    # Stop microphone input to avoid feedback
                    stream.stop_stream()

                    # Play the TTS response
                    # TODO: Format the text response so that it is appropriate for the spoken language.
                    vs.play_text_to_speech(output)
                    # cs.play_text_to_speech(output)

                    # Restart microphone input after response completes
                    stream.start_stream()



    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()