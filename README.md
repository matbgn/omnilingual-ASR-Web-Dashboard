<div align="center">
  <img src="./omniASR_header.jpg" alt="Omnilingual ASR Header" width="100%" />
</div>

# 🎙️ Omnilingual ASR Dashboard

Omnilingual ASR is an open-source speech recognition system supporting over **1,600 languages**. This repository contains a professional web-based dashboard for transcribing audio files, managing transcription history, and contributing to corpus creation.

---

## 🚀 Quick Start (Running the Dashboard)

If you have already performed the installation steps, you can start the dashboard using the dedicated virtual environment:

```bash
# From the project root
./asr_venv/bin/python app.py
```

- **URL:** [http://192.168.88.252:5000](http://192.168.88.252:5000)
- **Host:** Configured for `192.168.88.252` to allow local network access.

---

## 🛠️ Installation & Setup (Ubuntu 24.04 Optimized)

This project is optimized for Ubuntu 24.04 using a specific `venv` strategy to avoid C++ ABI conflicts and dependency overlaps.

### 1. Prerequisites
Ensure you have Python 3.10 and the necessary VENV tools installed:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10-venv ffmpeg
```

### 2. Environment Initialization
```bash
python3.10 -m venv asr_venv
./asr_venv/bin/pip install --upgrade pip
./asr_venv/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
./asr_venv/bin/pip install fairseq2==0.6 fairseq2n==0.6 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
./asr_venv/bin/pip install -r requirements-dashboard.txt
./asr_venv/bin/pip install omnilingual-asr --no-deps
```

---

## 📁 Model Management

The dashboard uses high-performance local checkpoints stored at `/mnt/TransferLearning/asr/asr`.

### Available Models
| Model Card | Parameters | Arch | Recommended Use |
|------------|------------|------|-----------------|
| `omniASR_LLM_1B_local` | 1B | LLM | **Default** - General purpose |
| `omniASR_CTC_1B_local` | 1B | CTC | High speed transcription |
| `omniASR_LLM_3B_local` | 3B | LLM | High accuracy |

### Local Configuration
Model cards are mapped via Fairseq2 asset cards. Your configuration is located at:
`~/.config/fairseq2/assets/cards/models/omniasr_local.yaml`

Example entry:
```yaml
name: omniASR_LLM_1B_local
model_family: wav2vec2_llama
model_arch: 1b
tokenizer_ref: omniASR_tokenizer_local
checkpoint: file:///mnt/TransferLearning/asr/asr/omniASR-LLM-1B.pt
```

---

## ⚠️ Troubleshooting & Important Notes

### 🎤 Microphone Access
Modern browsers require **HTTPS** or **localhost** to use the microphone (`getUserMedia`). 
- **Internal Access:** Use an SSH tunnel: `ssh -L 5000:localhost:5000 user@192.168.88.252` then visit `http://localhost:5000`.
- **Browser Flag:** In Chrome, enable `#unsafely-treat-insecure-origin-as-secure` for `http://192.168.88.252:5000`.

### ⚡ FFmpeg Libraries
If you encounter errors related to `libnppig.so.11`, ensure you are using the system-standard FFmpeg (`/usr/bin/ffmpeg`) and that `nvidia-npp-cu12` is installed in the venv.

### 🕰️ Audio Length
Inference is currently optimized for clips shorter than **40 seconds**. Long files are automatically chunked by the dashboard for processing.

---

## 📜 Credits & Citation

Omnilingual ASR is a research project by Meta AI.
```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={Omnilingual ASR Team},
    year={2025},
    url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/},
}
```
