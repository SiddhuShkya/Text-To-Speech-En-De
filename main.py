import os
import torch
import requests
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not API_TOKEN:
    raise ValueError("❌ Missing HUGGINGFACEHUB_API_TOKEN in your .env file!")

# CUDA check
if torch.cuda.is_available():
    print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not available — using CPU")

# ✅ Working translation model
API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-de"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def translate_en_to_de(text: str) -> str:
    """Translate English → German using Hugging Face Inference API."""
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    data = response.json()
    if isinstance(data, list) and "translation_text" in data[0]:
        return data[0]["translation_text"]
    return str(data)


if __name__ == "__main__":
    english_text = input("Enter English text to translate into German: ").strip()
    print("⏳ Translating via Hugging Face API...")
    german_text = translate_en_to_de(english_text)
    print("\n🇩🇪 German Translation:\n", german_text)
