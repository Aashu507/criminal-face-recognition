# 🔍 Criminal Face Recognition System

A locally-running, GPU-accelerated facial recognition pipeline for criminal identification, optimized for Indian face demographics.

## ⚡ Features

- **State-of-the-art accuracy** — InsightFace ArcFace (w600k_r50) embeddings
- **GPU-accelerated** — ONNX Runtime on NVIDIA RTX GPUs
- **Lighting normalization** — CLAHE preprocessing preserves skin tones
- **Persistent database** — ChromaDB vector store for fast 1:N search
- **Quality assessment** — Automatic image quality scoring and warnings
- **CLI tools** — Easy enrollment and search from command line
- **Webcam support** — Live camera/CCTV face search
- **Rich output** — Beautiful terminal tables and progress indicators

## 🛠️ Requirements

- **Python** 3.10+
- **NVIDIA GPU** with CUDA support (tested on RTX 5050)
- **CUDA Toolkit** 11.x or 12.x
- 24 GB RAM (recommended)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional — defaults work great)
```

### 3. Enroll Criminal Faces

```bash
# Enroll from a directory of images (filename = criminal ID):
python scripts/enroll_faces.py --dir data/criminals/

# Enroll a single face with metadata:
python scripts/enroll_faces.py --image mugshot.jpg --id CRIM001 --name "Suspect A" --case "CASE-2024-001"
```

### 4. Search for Matches

```bash
# Search with an image:
python scripts/search_face.py --image suspect_photo.jpg

# Search with custom threshold and top-K:
python scripts/search_face.py --image photo.jpg --threshold 0.40 --top-k 10

# Search with live webcam:
python scripts/search_face.py --webcam

# Export results to JSON:
python scripts/search_face.py --image photo.jpg --export results.json
```

## 📁 Project Structure

```
face-recognition/
├── core/
│   ├── preprocessor.py    # CLAHE enhancement + quality assessment
│   ├── detector.py        # SCRFD face detection (InsightFace)
│   ├── recognizer.py      # ArcFace embedding extraction
│   ├── database.py        # ChromaDB vector store
│   └── matcher.py         # End-to-end matching pipeline
├── scripts/
│   ├── enroll_faces.py    # Batch enrollment CLI
│   └── search_face.py     # Face search CLI
├── data/
│   ├── criminals/         # Place criminal mugshots here
│   └── queries/           # Place query images here
├── tests/
│   └── test_pipeline.py   # Unit and integration tests
├── requirements.txt
├── .env.example
└── README.md
```

## 🎯 Similarity Thresholds

| Threshold | Meaning                 | Use Case                    |
|-----------|-------------------------|-----------------------------|
| > 0.55    | ★ High confidence match | Automated alerts            |
| > 0.45    | Likely same person      | Default (recommended)       |
| > 0.35    | Possible match          | Broader search, manual review |
| < 0.30    | Different people        | False positive              |

## 🧪 Running Tests

```bash
# Run all tests (no GPU required):
python -m pytest tests/ -v

# Run only unit tests (fast, no model download):
python -m pytest tests/test_pipeline.py -v -k "not integration"
```

## ⚙️ Configuration (.env)

| Variable               | Default      | Description                      |
|------------------------|--------------|----------------------------------|
| `SIMILARITY_THRESHOLD` | `0.45`       | Minimum match similarity (0-1)   |
| `MODEL_NAME`           | `buffalo_l`  | InsightFace model pack           |
| `GPU_ID`               | `0`          | GPU device (-1 for CPU)          |
| `DET_SIZE`             | `640`        | Detection input resolution       |
| `TOP_K`                | `5`          | Max results per search           |
| `CHROMA_DB_DIR`        | `./chroma_db`| Database storage path            |

## 📜 Legal Notice

Facial recognition for law enforcement is subject to evolving regulation in India. Ensure compliance with the Information Technology Act and applicable state-level guidelines before deploying this system in production.

## 📄 License

For internal/research use only. See your organization's data handling policies.
