from gtts import gTTS
from playsound import playsound
import threading
import os

# Path to save temporary audio
TEMP_AUDIO = "temp_audio.mp3"

def speak(message):
    def _speak():
        try:
            tts = gTTS(text=message, lang='en', slow=False)
            tts.save(TEMP_AUDIO)  # Save to a known path
            playsound(TEMP_AUDIO)
            os.remove(TEMP_AUDIO)  # Delete after playing
        except Exception as e:
            print("TTS Error:", e)

    threading.Thread(target=_speak, daemon=True).start()
