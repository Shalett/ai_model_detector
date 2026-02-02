import zipfile
import tempfile
import os
import re
import json
import sys
from urllib.parse import urlparse

MODEL_EXTS = (".pt", ".pth", ".bin", ".ckpt", ".onnx", ".safetensors")
HF_BASE = "https://huggingface.co/"

PATTERNS = {
    "huggingface": [
        r'from_pretrained\(["\']([^"\']+)["\']\)',
        r'snapshot_download\([^)]*repo_id\s*=\s*["\']([^"\']+)["\']',
        r'DiffusionPipeline\.from_pretrained\(["\']([^"\']+)["\']'
    ],
    "torch_hub": [
        r'torch\.hub\.load\(["\']([^"\']+)["\'],\s*["\']([^"\']+)["\']'
    ],
    "timm": [
        r'create_model\(["\']([^"\']+)["\'].*pretrained\s*=\s*True'
    ],
    "urls": [
        r'(https?://[^\s"\']+)'
    ]
}

def extract(zip_path):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)
    return tmp

def is_model_url(url):
    return url.endswith(MODEL_EXTS)

def scan(root):
    results = []

    for r, _, files in os.walk(root):
        for f in files:
            path = os.path.join(r, f)

            # scan source & config files
            if f.endswith((".py", ".ipynb", ".json", ".yaml", ".yml", ".md")):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as file:
                        txt = file.read()

                        # Hugging Face
                        for p in PATTERNS["huggingface"]:
                            for m in re.findall(p, txt):
                                if "/" in m:
                                    results.append({
                                        "type": "huggingface",
                                        "id": m,
                                        "url": HF_BASE + m
                                    })

                        # Torch Hub
                        for repo, model in re.findall(PATTERNS["torch_hub"][0], txt):
                            results.append({
                                "type": "torch_hub",
                                "id": f"{repo}:{model}",
                                "url": f"https://github.com/{repo}"
                            })

                        # timm
                        for m in re.findall(PATTERNS["timm"][0], txt):
                            results.append({
                                "type": "timm",
                                "id": m,
                                "url": "https://github.com/rwightman/pytorch-image-models"
                            })

                        # direct URLs
                        for url in re.findall(PATTERNS["urls"][0], txt):
                            if is_model_url(url):
                                results.append({
                                    "type": "direct_download",
                                    "id": os.path.basename(urlparse(url).path),
                                    "url": url
                                })

                except Exception:
                    pass

            # Hugging Face config.json
            if f == "config.json":
                try:
                    with open(path) as cfg:
                        j = json.load(cfg)
                        if "_name_or_path" in j and "/" in j["_name_or_path"]:
                            m = j["_name_or_path"]
                            results.append({
                                "type": "huggingface",
                                "id": m,
                                "url": HF_BASE + m
                            })
                except:
                    pass

            # local checkpoints
            if f.endswith(MODEL_EXTS):
                results.append({
                    "type": "local_checkpoint",
                    "id": f,
                    "url": "local-file"
                })

    # deduplicate
    unique = { (r["type"], r["id"]): r for r in results }
    return list(unique.values())

def main(zip_file):
    print("\n📦 AI Model Dependency Scan (Universal)\n")
    root = extract(zip_file)
    results = scan(root)

    if not results:
        print("No pretrained model dependencies detected.")
        print("This project likely trains a model from scratch.\n")
        return

    for r in results:
        print(f"- Type      : {r['type']}")
        print(f"  Model ID  : {r['id']}")
        print(f"  Source    : {r['url']}\n")

    print("✔ Scan complete (AIBOM-safe).\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python universal_model_detector.py <project.zip>")
        sys.exit(1)

    main(sys.argv[1])
