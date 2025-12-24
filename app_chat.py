import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# ------------------ VOICE MODULES -------------------
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
from gtts import gTTS
import tempfile
# ----------------------------------------------------

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="Gemini Voice Chat", layout="wide")

new_chat_id = f"{time.time()}"
MODEL_ROLE = "ai"
AI_AVATAR_ICON = "✨"

# Create data folder
os.makedirs("data", exist_ok=True)

# Load past chats
try:
    past_chats = joblib.load("data/past_chats_list")
except:
    past_chats = {}

# Sidebar
with st.sidebar:
    st.write("## Past Chats")

    st.session_state.chat_id = st.selectbox(
        "Pick a chat:",
        [new_chat_id] + list(past_chats.keys()),
        index=0,
    )

    st.session_state.chat_title = f"ChatSession-{st.session_state.chat_id}"

st.write("## Chat with Gemini")

# Load chat history
try:
    st.session_state.messages = joblib.load(f"data/{st.session_state.chat_id}-st_messages")
    st.session_state.gemini_history = joblib.load(f"data/{st.session_state.chat_id}-gemini_messages")
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []

# Gemini model
st.session_state.model = genai.GenerativeModel("gemini-2.0-flash-001")
st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])


# ------------------ VOICE INPUT FUNCTIONS -------------------
def record_voice(duration=5):
    """Record microphone audio and save as proper PCM WAV."""
    st.info("🎤 Listening... Speak now...")

    fs = 44100
    # FIX → Use PCM 16 bit WAV (SpeechRecognition requirement)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    filepath = "voice_input.wav"

    # ensure proper PCM format
    audio = audio.reshape(-1).astype("int16")
    write(filepath, fs, audio)

    return filepath


def speech_to_text():
    filepath = record_voice()

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filepath) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: **{text}**")
        return text
    except Exception as e:
        st.error(f"❌ Could not understand audio: {e}")
        return ""


def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
# -------------------------------------------------------------


# ------------------ MIC BUTTON (Always visible) -------------------
st.markdown("###")

mic_col, _ = st.columns([1, 5])
with mic_col:
    mic_clicked = st.button("🎤 Speak")

if mic_clicked:
    spoken_text = speech_to_text()
    st.session_state.voice_text = spoken_text
else:
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = ""


# ------------------ CHAT INPUT -------------------
prompt = st.chat_input("Type or speak your message...")

# If spoken text exists → override typed text
if st.session_state.voice_text:
    prompt = st.session_state.voice_text
    st.session_state.voice_text = ""


# ------------------ PROCESS MESSAGE -------------------
if prompt:
    # Save chat session title
    if st.session_state.chat_id not in past_chats:
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, "data/past_chats_list")

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Gemini Response (Streaming)
    response = st.session_state.chat.send_message(prompt, stream=True)

    with st.chat_message(MODEL_ROLE, avatar=AI_AVATAR_ICON):
        full_msg = ""
        placeholder = st.empty()

        for chunk in response:
            if hasattr(chunk, "text"):
                full_msg += chunk.text
                placeholder.write(full_msg + "▌")

        placeholder.write(full_msg)

    # Save AI output
    st.session_state.messages.append(
        {
            "role": MODEL_ROLE,
            "content": st.session_state.chat.history[-1].parts[0].text,
            "avatar": AI_AVATAR_ICON,
        }
    )

    # Save chats
    joblib.dump(st.session_state.messages, f"data/{st.session_state.chat_id}-st_messages")
    joblib.dump(st.session_state.chat.history, f"data/{st.session_state.chat_id}-gemini_messages")

    # Speak response
    speak_text(full_msg)
