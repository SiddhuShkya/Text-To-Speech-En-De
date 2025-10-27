import os
import re
import shutil
import torch
import streamlit as st  # type: ignore
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # type: ignore
from gtts import gTTS  # type: ignore
import tempfile

# -------------------------------
# 🔹 Optional: Reduce CUDA fragmentation
# -------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------------
# 🔹 Ensure output folder exists
# -------------------------------
OUTPUT_DIR = "audios"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------
# 🔹 Sanitize filenames
# -------------------------------
def sanitize_filename(name: str) -> str:
    """Replace unsafe characters in filenames with underscores."""
    name = name.strip().lower()
    # Only allow letters, numbers, dash, underscore
    return re.sub(r"[^a-z0-9_\-]", "_", name)


# -------------------------------
# 🔹 Helper: List saved audio files
# -------------------------------
def list_saved_audios():
    """Return sorted list of saved .mp3 files."""
    st.session_state.saved_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp3")]
    )
    return st.session_state.saved_files


# -------------------------------
# 🔹 Load translation model once per session
# -------------------------------
def get_translation_model():
    """Load translation model & tokenizer once per session."""
    if "tokenizer" not in st.session_state or "model" not in st.session_state:
        model_name = "facebook/m2m100_418M"
        st.session_state.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        st.session_state.model = M2M100ForConditionalGeneration.from_pretrained(
            model_name
        )

        # Use CUDA if available
        if torch.cuda.is_available():
            st.session_state.device = "cuda"
            st.session_state.model.to("cuda")
        else:
            st.session_state.device = "cpu"
            st.warning("⚠️ CUDA not available — using CPU")

    return st.session_state.tokenizer, st.session_state.model, st.session_state.device


# -------------------------------
# 🔹 Translation function (English → German)
# -------------------------------
def translate_en_to_de(text: str, tokenizer, model, device: str) -> str:
    """Translate English text to German using M2M100 model."""
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
    """Convert German text to speech and return temporary file path."""
    tts = gTTS(text=text, lang="de")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_file)
    return temp_file


# -------------------------------
# 🔹 Generate safe filename
# -------------------------------
def get_audio_filename(english_text: str) -> str:
    words = english_text.strip().lower().split()[:8]
    base_name = "_".join(words)
    return f"{sanitize_filename(base_name)}.mp3"


# -------------------------------
# 🔹 Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="English → German + Speech", layout="wide")
    st.header("🇬🇧 English → 🇩🇪 German")
    st.write("---")

    # -------------------------------
    # 🧹 Handle pending file deletion (safe cleanup)
    # -------------------------------
    if "delete_pending" in st.session_state:
        try:
            os.remove(st.session_state.delete_pending)
            st.toast(f"🗑️ Deleted: {os.path.basename(st.session_state.delete_pending)}")
        except FileNotFoundError:
            pass
        del st.session_state.delete_pending
        st.session_state.saved_files = list_saved_audios()

    # -------------------------------
    # 💅 Styling
    # -------------------------------
    st.markdown(
        """
        <style>
        textarea {
            font-size: 20px !important;
            line-height: 1.6 !important;
            font-family: 'Segoe UI', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------
    # 🔹 Load model (once per session)
    # -------------------------------
    tokenizer, model, device = get_translation_model()

    # -------------------------------
    # 🔹 Layout (Input ↔ Output)
    # -------------------------------
    col1, col2 = st.columns([5, 5], gap="small")
    with col1:
        with st.container(border=True):
            english_text = st.text_area(
                "Enter English Text:",
                placeholder="Type your English sentence here...",
                height=150,
                key="input_text",
            )
        translate_button = st.button("Translate", use_container_width=True)

    # -------------------------------
    # 🔹 Translation + Speech
    # -------------------------------
    if translate_button and english_text.strip():
        with st.spinner("⏳ Translating..."):
            st.session_state.translated_text = translate_en_to_de(
                english_text, tokenizer, model, device
            )

        with st.spinner("🎧 Generating German speech..."):
            st.session_state.audio_path = generate_german_audio(
                st.session_state.translated_text
            )

    # -------------------------------
    # 🔹 Display Translation and Audio
    # -------------------------------
    with col2:
        german_text = st.session_state.get(
            "translated_text", "The translated text will appear here."
        )
        with st.container(border=True):
            st.text_area(
                "🇩🇪 German Translation:", value=german_text, height=150, disabled=True
            )

        if "audio_path" in st.session_state:
            audio_col1, audio_col2 = st.columns([19, 3])
            with audio_col1:
                st.audio(st.session_state.audio_path, format="audio/mp3")
            with audio_col2:
                if st.button("💾 Save Audio"):
                    filename = get_audio_filename(english_text)
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    shutil.copy(st.session_state.audio_path, save_path)
                    st.toast(f"✅ Audio saved as `{filename}`")
                    st.session_state.saved_files = list_saved_audios()

    # -------------------------------
    # 🔹 Saved Audios Section
    # -------------------------------
    saved_files = list_saved_audios()
    st.write("---")
    c1, c2 = st.columns([5, 5], gap="small")

    # 🔸 Left Panel — Single Select + Play
    with c2:
        if saved_files:
            st.write("#### 📂 :green[Search Saved Audio]")
            options = ["Select an audio"] + [
                f.replace(".mp3", "").replace("_", " ").title() for f in saved_files
            ]
            with st.container(border=True):
                selected_audio = st.selectbox(
                    "Select Saved Audio",
                    options=options,
                    label_visibility="collapsed",  # hides label visually but keeps it accessible
                )

            if selected_audio != "Select an audio":
                matched_file = [
                    f
                    for f in saved_files
                    if selected_audio.lower().replace(" ", "_") in f.lower()
                ]
                if matched_file:
                    file_path = os.path.join(OUTPUT_DIR, matched_file[0])
                    with st.container(border=True):
                        audio_col1, audio_col2 = st.columns([10, 2])
                        with audio_col1:
                            st.audio(file_path, format="audio/mp3")
                        with audio_col2:
                            if st.button(
                                "🗑️ Delete Audio", key=f"del_{matched_file[0]}"
                            ):
                                st.session_state.delete_pending = file_path
                                st.rerun()

    # 🔸 Right Panel — List All Saved Files
    with c1:
        if saved_files:
            st.write("#### 📂 :green[Saved German Audios]")
            with st.container(border=True, height=400):
                for idx, f in enumerate(saved_files):
                    display_name = f.replace(".mp3", "").replace("_", " ").title()
                    file_path = os.path.join(OUTPUT_DIR, f)
                    with st.expander(f"{display_name}"):
                        audio_col1, audio_col2 = st.columns([10, 2])
                        with audio_col1:
                            st.audio(file_path, format="audio/mp3")
                        with audio_col2:
                            if st.button("🗑️ Delete Audio", key=f"del_{idx}"):
                                st.session_state.delete_pending = file_path
                                st.rerun()


# -------------------------------
# 🔹 Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
