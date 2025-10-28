# 🗣️ English → German Text-to-Speech (TTS) App

> A web application that translates **English text into German** using the **Helsinki-NLP MarianMT model** and generates **German speech** via **gTTS**. The system provides **interactive text display**, **audio playback**, and allows users to **save or delete generated audio files**. Supports **GPU acceleration** with Docker for faster translation.

---

## 📌 What It Does

- Translates English sentences into **natural, idiomatic German**.
- Generates **German speech audio (MP3)** from the translated text.
- Allows playback of generated audio inside the app.
- Save audio files for later use or delete them.
- Optional GPU support for faster translations using **PyTorch + CUDA**.
- Dockerized for easy deployment and reproducibility.
- Optional persistent audio storage via Docker volume.

---

## 🛠️ Technology Stack

| Category            | Tools |
|---------------------|-------|
| **Programming**     | Python 3.12 |
| **Framework**       | Streamlit |
| **Machine Learning**| Transformers (Helsinki-NLP MarianMT) |
| **Text-to-Speech**  | gTTS |
| **Audio Handling**  | tempfile, shutil |
| **Containerization**| Docker (CPU/GPU) |

---

## 🔥 Key Features

- ✅ **Accurate Translation** – English → German using Helsinki-NLP MarianMT for natural, idiomatic results.
- 🎧 **Speech Synthesis** – Generate German MP3 audio via gTTS.
- 💾 **Save & Delete Audio** – Manage audio files within the app.
- 🖥️ **GPU Acceleration** – Optional PyTorch + CUDA for faster inference.
- 🐳 **Dockerized** – Easy deployment across systems.
- 🗂️ **Persistent Storage** – Save audios using Docker volume mapping.
- 💅 **Streamlit UI** – Interactive and responsive user interface.

---

## 📂 Project Structure

```
📂 Text-To-Speech-En-De/
├── 📄 main.py           # Streamlit application
├── 📄 requirements.txt  # Python dependencies
├── 📂 audios/           # Optional: saved MP3 files (persist via volume)
├── 📄 Dockerfile        # Docker configuration (CPU/GPU)
├── 📄 .dockerignore     # Files/Folders ignored by Docker
├── 📄 README.md         # Project documentation
├── 📂 .streamlit/       # Contains config.toml for app theme
└── 📂 venv/             # Virtual environment (ignored in Docker)
```

---

## 🧭 How to Run Locally (CPU)

### **1. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the App**

```bash
streamlit run main.py
```

### **4. Open in Browser**

```bash
http://localhost:8501
```

--- 

## 🐳 Run with Docker

### **1. Build the Docker Image**

```bash
sudo docker build -t tts-en-de-cpu .
```

### **2. Run the container on CPU with persistent audio storage**

```bash
sudo docker run -p 8501:8501 -v $(pwd)/audios:/app/audios tts-en-de-cpu
```
- `-p 8501:8501` → maps Streamlit port to your host.
- `-v $(pwd)/audios:/app/audios` → keeps generated MP3s even after the container stops.

### **3. Open in Browser**

```bash
http://localhost:8501
```

