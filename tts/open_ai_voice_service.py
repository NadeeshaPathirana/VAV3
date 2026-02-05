import simpleaudio as sa
from pathlib import Path
from openai import OpenAI
import os

# Initialize client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Output file path
speech_file_path = Path("output.wav")

# Request speech synthesis
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",   # OpenAI TTS model
    voice="alloy",             # voice options: alloy, verse, etc.
    input="Hello! This is OpenAI speaking."
) as response:
    response.stream_to_file(speech_file_path)

# Play the audio
wave_obj = sa.WaveObject.from_wave_file(str(speech_file_path))
play_obj = wave_obj.play()
play_obj.wait_done()
