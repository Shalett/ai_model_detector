# AI Model Detector

AI Model Detector is a lightweight tool that scans AI/ML projects and identifies
**pretrained model dependencies**. It helps answer the question:

> “What pretrained models does this project depend on, and where do they come from?”

This is especially useful for building an **AI Bill of Materials (AIBOM)**,
performing model provenance checks, or auditing third-party AI repositories.

---

## ✨ Features

- Detects **Hugging Face** model IDs and generates source links
- Detects models loaded from **Torch Hub**
- Detects **timm** vision models (`pretrained=True`)
- Detects **direct model downloads** (URLs ending in `.pt`, `.bin`, `.onnx`, etc.)
- Detects **local checkpoint files**
- Works on **any zipped GitHub project**
- No external Python dependencies (standard library only)

---

## 📦 How It Works

The tool statically scans files inside a ZIP archive and looks for common model-loading
patterns, including:

- `from_pretrained(...)`
- `snapshot_download(...)`
- `DiffusionPipeline.from_pretrained(...)`
- `torch.hub.load(...)`
- `timm.create_model(..., pretrained=True)`
- Direct URLs pointing to model files
- Local model checkpoint files

It then outputs a **deduplicated list of detected model dependencies** along with
their inferred source.

---

## 🚀 Usage

### 1. Clone the repository


git clone https://github.com/Shalett/ai_model_detector.git
cd ai_model_detector

2. Place a ZIP file of the AI project in the directory
Example:

sentence_transformers.zip

3. Run the detector
python universal_model_detector.py sentence_transformers.zip
📄 Example Output
📦 AI Model Dependency Scan (Universal)

- Type      : huggingface
  Model ID  : sentence-transformers/all-MiniLM-L6-v2
  Source    : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

✔ Scan complete (AIBOM-safe).
