import os
import json
import uuid
import re
import argparse
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict

import requests
import certifi
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util import ssl_

# ─────────────────────────────────────────────────────────────
# SSL FIX (Corporate-safe)
# ─────────────────────────────────────────────────────────────
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl_.create_urllib3_context()
        ctx.load_default_certs()
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

SESSION = requests.Session()
SESSION.mount("https://", SSLAdapter())
SESSION.verify = certifi.where()

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
HF_MODEL_API = "https://huggingface.co/api/models/{}"
HF_DATASET_API = "https://huggingface.co/api/datasets/{}"
HF_DATASET_URL = "https://huggingface.co/datasets/{}"
HF_MODEL_README = "https://huggingface.co/{}/raw/main/README.md"

# ─────────────────────────────────────────────────────────────
# BASIC HELPERS
# ─────────────────────────────────────────────────────────────
def bom_ref(name: str) -> str:
    return f"{name}-{uuid.uuid5(uuid.NAMESPACE_OID, name)}"

def prop(name: str, value: str) -> dict:
    return {"name": name, "value": value}

def add_ext_ref(comp: dict, url: str, typ: str = "documentation"):
    comp.setdefault("externalReferences", []).append({"url": url, "type": typ})

def new_bom() -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {}
    }

# ─────────────────────────────────────────────────────────────
# HTTP HELPERS (SAFE)
# ─────────────────────────────────────────────────────────────
def get_json(url: str, token: Optional[str] = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = SESSION.get(url, headers=headers, timeout=30)
        if r.status_code == 401:
            return {}
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def get_text(url: str) -> str:
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────
# SPDX NORMALIZATION (DETERMINISTIC)
# ─────────────────────────────────────────────────────────────
def spdx_lookup(raw: Optional[str]):
    if not raw:
        return None
    lic = raw.strip().upper()
    return {
        "id": lic,
        "url": f"https://spdx.org/licenses/{lic}.html"
    }

# ─────────────────────────────────────────────────────────────
# DATASET ID VALIDATION
# ─────────────────────────────────────────────────────────────
HF_DATASET_ID = re.compile(r"^[a-zA-Z0-9][\w\-]+(/[a-zA-Z0-9][\w\-]+)?$")

def is_valid_dataset_id(name: str) -> bool:
    return bool(HF_DATASET_ID.match(name.strip()))


# ⬇️ ADD IT HERE
def extract_training_section_datasets(md: str) -> List[Dict[str, Optional[str]]]:
    """
    Extract datasets from README training section.
    Returns [{name, url}]
    """
    if not md:
        return []

    lines = md.splitlines()
    capture = False
    datasets = []

    for line in lines:
        l = line.strip()

        if re.match(r"#+\s*training datasets?", l.lower()):
            capture = True
            continue

        if capture and l.startswith("#"):
            break

        if capture and l.startswith(("-", "*")):
            # Markdown link
            m = re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", l)
            if m:
                name = m.group(1).strip()
                url = m.group(2).strip()
                datasets.append({"name": name, "url": url})
                continue

            # Plain text fallback
            name = re.sub(r"^[\-\*\s]+", "", l)
            name = re.sub(r"\s*\(.*?\)\s*$", "", name)
            if name:
                datasets.append({"name": name.strip(), "url": None})

    return datasets


# ─────────────────────────────────────────────────────────────
# DATASET EXTRACTION (OPTION A)
# ─────────────────────────────────────────────────────────────
def extract_datasets(model_api: dict, model_id: str) -> Dict[str, Dict]:
    """
    Returns:
    datasets[ds] = {
    "kind": "hf",
    "id": ds,
    "name": ds,
    "url": HF_DATASET_URL.format(ds)
     }

    """
    datasets: Dict[str, Dict] = {}

    # 1️⃣ Trust model card (authoritative)
    card_datasets = model_api.get("cardData", {}).get("datasets", [])
    if isinstance(card_datasets, str):
        card_datasets = [card_datasets]

    for ds in card_datasets:
        if isinstance(ds, str) and is_valid_dataset_id(ds):
            datasets[ds] = {
                "kind": "hf",
                "id": ds,
                "url": HF_DATASET_URL.format(ds)
            }

    # 2️⃣ README training section (augment)
    md = get_text(HF_MODEL_README.format(model_id))
    for entry in extract_training_section_datasets(md):
        name = entry["name"]
        url = entry.get("url")

        # HF dataset link detected
        hf_match = re.search(r"huggingface\.co/datasets/([\w\-\/]+)", url or "")
        if hf_match:
            ds_id = hf_match.group(1)
            datasets[ds_id] = {
                "kind": "hf",
                "id": ds_id,
                "url": HF_DATASET_URL.format(ds_id)
            }
            continue

        # Non-HF dataset
        if name not in datasets:
            datasets[name] = {
            "kind": "non-hf",
            "id": None,
            "name": name,
            "url": url
            }


    return datasets



# ─────────────────────────────────────────────────────────────
# DATASET COMPONENT
# ─────────────────────────────────────────────────────────────
def dataset_component(ds: Dict, token: Optional[str]) -> dict:
    name = ds.get("id") or ds.get("name")


    comp = {
        "type": "data",
        "bom-ref": bom_ref(f"dataset::{name}"),
        "name": name,
        "properties": [
            prop("usage", "training"),
            prop("dataset_type", ds["kind"])
        ]
    }

    # URL
    if ds.get("url"):
        add_ext_ref(comp, ds["url"])

    # HF license resolution
    if ds["kind"] == "hf" and ds.get("id"):
        api = get_json(HF_DATASET_API.format(ds["id"]), token)
        lic = (api.get("cardData") or {}).get("license")
        if isinstance(lic, list) and lic:
            lic = lic[0]

        if lic:
            spdx = spdx_lookup(str(lic))
            if spdx:
                comp["properties"].append(prop("license_spdx_id", spdx["id"]))
                comp["properties"].append(prop("license_spdx_url", spdx["url"]))
            else:
                comp["properties"].append(prop("license", str(lic)))

    if ds["kind"] == "non-hf":
        comp["properties"].append(
            prop("note", "non-huggingface dataset reference")
        )

    return comp



# ─────────────────────────────────────────────────────────────
# MODEL COMPONENT
# ─────────────────────────────────────────────────────────────
def model_component(model_id: str, token: Optional[str]):
    api = get_json(HF_MODEL_API.format(model_id), token)
    md = get_text(HF_MODEL_README.format(model_id))

    comp = {
        "type": "machine-learning-model",
        "bom-ref": bom_ref(model_id),
        "name": model_id,
        "externalReferences": [
            {"url": f"https://huggingface.co/{model_id}", "type": "documentation"}
        ],
        "modelCard": {"modelParameters": {}}
    }

    # Author
    if api.get("author"):
        comp["authors"] = [{"name": api["author"]}]

    # Task
    if api.get("pipeline_tag"):
        comp["modelCard"]["modelParameters"]["task"] = api["pipeline_tag"]

    # License
    lic = (api.get("cardData") or {}).get("license")
    if isinstance(lic, list) and lic:
        lic = lic[0]

    if lic:
        spdx = spdx_lookup(str(lic))
        if spdx:
            comp["licenses"] = [{
                "license": {
                    "id": spdx["id"],
                    "url": spdx["url"]
                }
            }]

    # Datasets (Option A)
    dataset_map = extract_datasets(api, model_id)

    ds_components = [
    dataset_component(ds, token)
    for ds in dataset_map.values()
    ]

    if dataset_map:
     comp["modelCard"]["modelParameters"]["datasets"] = list(dataset_map.keys())


    # ✅ REQUIRED RETURN
    return comp, ds_components


# ─────────────────────────────────────────────────────────────
# AIBOM GENERATION
# ─────────────────────────────────────────────────────────────
def generate_aibom(model_id: str, token: Optional[str]) -> dict:
    model, datasets = model_component(model_id, token)

    bom = new_bom()
    bom["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    bom["metadata"]["component"] = model

    if datasets:
        bom["components"] = datasets
        bom["dependencies"] = [{
            "ref": model["bom-ref"],
            "dependsOn": [d["bom-ref"] for d in datasets]
        }]

    return bom

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Generate CycloneDX AI-BOM (clean & deterministic)")
    parser.add_argument("model_id")
    parser.add_argument("-o", "--output")
    parser.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_TOKEN"))
    args = parser.parse_args()

    bom = generate_aibom(args.model_id, args.hf_token)

    out = args.output or f"{args.model_id.replace('/', '_')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bom, f, indent=2)

    print(f"✅ Clean AIBOM generated: {out}")

if __name__ == "__main__":
    main()
