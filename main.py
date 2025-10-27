import os
import shutil
import torch
import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from gtts import gTTS
import tempfile

# Optional: Reduce CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure output folder exists
OUTPUT_DIR = "audios"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------
# 🔹 Load translation model once per session
# -------------------------------
def get_translation_model():
    if "tokenizer" not in st.session_state or "model" not in st.session_state:
        model_name = "facebook/m2m100_418M"
        st.session_state.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        st.session_state.model = M2M100ForConditionalGeneration.from_pretrained(
            model_name
        )

        if torch.cuda.is_available():
            st.session_state.device = "cuda"
            st.session_state.model.to("cuda")
            st.success(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
        else:
            st.session_state.device = "cpu"
            st.warning("⚠️ CUDA not available — using CPU")

    return st.session_state.tokenizer, st.session_state.model, st.session_state.device


# -------------------------------
# 🔹 Translation function
# -------------------------------
def translate_en_to_de(text: str, tokenizer, model, device: str) -> str:
    tokenizer.src_lang = "en"
    encoded = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id("de"),
            max_length=256,
        )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    torch.cuda.empty_cache()
    return translation


# -------------------------------
# 🔹 Generate German Speech with gTTS
# -------------------------------
def generate_german_audio(text: str) -> str:
    """Convert German text to speech (MP3) and return temp file path."""
    tts = gTTS(text=text, lang="de")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_file)
    return temp_file


# -------------------------------
# 🔹 Helper to create filename from first 5 English words
# -------------------------------
def get_audio_filename(english_text: str) -> str:
    words = english_text.strip().lower().split()
    words = words[:8]  # first 5 words
    filename = "_".join(words) + ".mp3"
    return filename


# -------------------------------
# 🔹 Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="English → German + Speech", layout="wide")
    st.title("🗣️ English → German Translator with Speech")
    st.write("---")

    # Custom styling
    st.markdown(
        """
        <style>
        textarea {
            font-size: 24px !important;
            line-height: 1.6 !important;
            font-family: 'Segoe UI', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load translation model
    tokenizer, model, device = get_translation_model()

    col1, col2, col3 = st.columns([5, 5, 5], gap="small")
    with col1:
        with st.container(border=True):
            english_text = st.text_area(
                "Enter English Text:",
                placeholder="Type your English sentence here...",
                height=250,
                key="input_text",
            )
        translate_button = st.button("Translate", use_container_width=True)

    if translate_button and english_text.strip():
        with st.spinner("⏳ Translating..."):
            german_text = translate_en_to_de(english_text, tokenizer, model, device)
            st.session_state["translated_text"] = german_text

        with st.spinner("🎧 Generating German speech..."):
            audio_path = generate_german_audio(german_text)
            st.session_state["audio_path"] = audio_path

    with col2:
        with st.container(border=True):
            german_text = st.session_state.get(
                "translated_text", "The translated text will appear here."
            )
            st.text_area(
                "🇩🇪 German Translation:", value=german_text, height=250, disabled=True
            )

        # Display audio player and Save button side-by-side
        if "audio_path" in st.session_state:
            audio_col1, audio_col2 = st.columns([8.5, 1.5])
            with audio_col1:
                st.audio(st.session_state["audio_path"], format="audio/mp3")
            with audio_col2:
                if st.button("💾 Audio"):
                    filename = get_audio_filename(english_text)
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    shutil.copy(st.session_state["audio_path"], save_path)
                    st.toast(f"✅ Audio saved to `{save_path}`")


# -------------------------------
# 🔹 Entry point
# -------------------------------
if __name__ == "__main__":
    main()
