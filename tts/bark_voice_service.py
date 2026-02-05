import os
import time
import pygame
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
import torch


def play_bark_tts(text, voice_preset="v2/en_speaker_1"):
    start_time = time.time()
    temp_audio_file = "../temp_bark_audio.wav"

    _old_load = torch.load

    def torch_load_wrapper(*args, **kwargs):
        kwargs["weights_only"] = False  # allow legacy model pickles
        return _old_load(*args, **kwargs)

    torch.load = torch_load_wrapper

    print("🔊 Generating audio with Bark...")
    audio_array = generate_audio(text, history_prompt=voice_preset)

    # Save to WAV file
    write_wav(temp_audio_file, SAMPLE_RATE, audio_array)
    print("✅ Audio generated and saved!")

    # Play audio with pygame
    pygame.mixer.init(frequency=SAMPLE_RATE)
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

    time.sleep(1)
    os.remove(temp_audio_file)

    end_time = time.time()
    print(f"Bark TTS Execution Time: {end_time - start_time:.2f} seconds")

play_bark_tts("Hi there, this is a test")