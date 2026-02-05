import os
import time
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

from tts import google_voice_service as vs
# from tts import coqui_voice_service as cs
from rag.AIVA import AIVA
from rag.AIVA_Chroma import AIVA_Chroma

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10

# ai_assistant = AIVoiceAssistant()
# ai_assistant = AIVA()
ai_assistant = AIVA_Chroma()

# V1 - before optimising recording process

def is_silence(data, threshold=500):
    """Check if audio data contains silence."""
    return np.max(np.abs(data)) < threshold


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    start_time = time.time()  # Start time measurement
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            end_time = time.time()  # End time measurement
            execution_time = end_time - start_time
            print(f"Record Audio Execution Time: {execution_time:.2f} seconds")  # Print the total execution time
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False


def transcribe_audio(model, file_path):
    start_time = time.time()
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    end_time = time.time()  # End time measurement
    execution_time = end_time - start_time
    print(f"Transcribe Audio Execution Time: {execution_time:.2f} seconds")  # Print the total execution time
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
            chunk_file = "temp_audio_chunk.wav"
            # Todo: allow user to interrupt while the VA is talking

            # Record audio chunk
            print("_")
            if not record_audio_chunk(audio, stream):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)

                # Add user input to transcript
                user_input_transcription += "User: " + transcription + "\n"
                # Process user input and get response from AI assistant
                print("User:{}".format(transcription))
                output = ai_assistant.interact_with_llm(transcription)
                if output:
                    output = output.lstrip()
                    print("AI Assistant:{}".format(output))

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