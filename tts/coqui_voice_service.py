import os
import time
import simpleaudio as sa
from TTS.api import TTS
import re

tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False) #model doesn't have a speaker manager
# tts = TTS(model_name="tts_models/en/ljspeech/speedy-speech", progress_bar=False) #model doesn't have a speaker manager
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False) #gives below messege
# Connected to pydev debugger (build 232.10203.26)
#  > You must confirm the following:
#  | > "I have purchased a commercial license from Coqui: licensing@coqui.ai"
#  | > "Otherwise, I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml" - [y/n]
#  | | >

def clean_text_for_tts(text: str) -> str:
    # Remove emojis and other non-ASCII symbols
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove excessive spaces (collapse multiple spaces → one space)
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def play_text_to_speech(text, emotion="neutral", pitch=1.0, speed=0.25):
    start_time = time.time()
    text = clean_text_for_tts(text)

    try:
        # Generate TTS to a temporary file
        tts.tts_to_file(text=text, file_path="temp_audio.wav")

        # Play the audio
        wave_obj = sa.WaveObject.from_wave_file("temp_audio.wav")
        end_time = time.time()
        print(f"TTS Execution Time: {end_time - start_time:.2f} seconds")
        play_obj = wave_obj.play()
        play_obj.wait_done()

        # Remove temp file
        os.remove("temp_audio.wav")

    except Exception as e:
        print(f"[ERROR] TTS failed for sentence {text}: {e}")
        answer = "I'm sorry, I had a little trouble replying to you just now — but I’m back!"
        play_text_to_speech(answer)


#
def play_text_to_speech_sentence_wise(text, emotion="neutral", pitch=1.0, speed=1.0):
    start_time = time.time()
    text = clean_text_for_tts(text)

    # Split long text into sentences safely
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    for i, sentence in enumerate(sentences, 1):
        # Skip sentences that are too short (less than ~10 chars) - to fix RuntimeError: Calculated padded input size per channel: (4). Kernel size: (5). Kernel size can't be greater than actual input size
        if len(sentence) < 10:
            print(f"[WARN] Skipping very short text: '{sentence}'")
            continue

        try:
            tts.tts_to_file(text=sentence, file_path="temp_audio.wav", speed=speed)
            wave_obj = sa.WaveObject.from_wave_file("temp_audio.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()
            os.remove("temp_audio.wav")
        except Exception as e:
            print(f"[ERROR] TTS failed for sentence {i}: {e}")
            continue

    end_time = time.time()
    print(f"TTS Execution Time: {end_time - start_time:.2f} seconds")


# play_text_to_speech("I am very sad right now.")
# print("models:", TTS().list_models())

# Get the full model dictionary
# tts_models_dict = TTS().list_models().models_dict
#
# # Iterate over all models
# for category, langs in tts_models_dict.items():
#     for lang, datasets in langs.items():
#         for dataset, models in datasets.items():
#             for model_name in models.keys():
#                 full_model_name = f"{category}/{lang}/{dataset}/{model_name}"
#                 print(f"\n🔹 Model: {full_model_name}")
#
#                 try:
#                     # Load model metadata (no full init to save time)
#                     tts = TTS(full_model_name)
#
#                     # Print speakers if available
#                     if hasattr(tts, "speakers") and tts.speakers:
#                         print(f"  🗣️ Speakers: {len(tts.speakers)} available")
#                         print("   ", ", ".join(tts.speakers[:10]), "..." if len(tts.speakers) > 10 else "")
#                     else:
#                         print("  🗣️ Single-speaker or no speaker info.")
#
#                     # Optional: print languages
#                     if hasattr(tts, "languages") and tts.languages:
#                         print(f"  🌐 Languages: {tts.languages}")
#
#                 except Exception as e:
#                     print(f"  ⚠️ Could not load model ({e})")

# this code gives below msg
#  > tts_models/multilingual/multi-dataset/xtts_v2 has been updated, clearing model cache...
#  > You must confirm the following:
#  | > "I have purchased a commercial license from Coqui: licensing@coqui.ai"
#  | > "Otherwise, I agree to the terms of the non-commercial CPML: https://coqui.ai/cpml" - [y/n]
#  | | >