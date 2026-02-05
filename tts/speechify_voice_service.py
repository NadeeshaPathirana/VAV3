from speechify import Speechify
from speechify.core.api_error import ApiError
import simpleaudio as sa
import base64
import time
from tts import google_voice_service as vsg

client = Speechify(
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjM5ODYzNTQsImlzcyI6InNwZWVjaGlmeS1hcGkiLCJzY29wZSI6ImF1ZGlvOmFsbCB2b2ljZXM6cmVhZCIsInN1YiI6IjR3aFJTT1cybEhOMTJTQjdxYkQ1OWhUelJ2ZjEifQ.i6QenJDQwb4vZsnXWgwpDSLvgJuomRXwH_1E5cHvmG4",
) # token expires in 1h


def get_emotion(emotion):
    if emotion == 'Happiness':
        emotion_req = 'cheerful'
    elif emotion == 'Anger':
        emotion_req = 'calm'
    elif emotion == 'Sadness':
        emotion_req = 'sad'
    else:
        emotion_req = ''
    return emotion_req


def play_text_to_speech(text, emotion):
    start_time = time.time()
    emotion_req = get_emotion(emotion)
    if emotion_req == '':
        inp = "<speak>" + text + "</speak>"
    else:
        print(emotion_req)
        inp = "<speak><speechify:style emotion=\"" + emotion_req + "\">" + text + "</speechify:style></speak>"
    try:
        response = client.tts.audio.speech(
            input=inp,
            voice_id="lisa",
            model="simba-english",
            # emotion='sad',
        )

        audio_bytes = base64.b64decode(response.audio_data)

        # Save audio file
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)

        end_time = time.time()
        # Play the WAV file
        wave_obj = sa.WaveObject.from_wave_file("output.wav")
        print(f"TTS Interaction Time: {end_time - start_time:.2f} seconds")

        play_obj = wave_obj.play()
        play_obj.wait_done()

    except ApiError as e:
        print(e.status_code)
        print(e.body)
        vsg.play_text_to_speech(text)  # for google


# play_text_to_speech("Hi there, are you alright", "Happiness") # Neutral => "", Happiness => "cheerful", Anger => "calm", Sadness => "sad"