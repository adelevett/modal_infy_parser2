"""
Modal deployment for Infinity-Parser2-Pro
=========================================
Usage:
  # One-shot parse (returns markdown string)
  modal run infinity_parser2_modal.py --input path/to/doc.pdf

  # Deploy as a persistent API endpoint
  modal deploy infinity_parser2_modal.py

Requirements:
  pip install modal
  modal setup          # authenticate once
"""

import os
import base64
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME        = "infinity-parser2"
MODEL_ID        = "infly/Infinity-Parser2-Pro"
MODEL_CACHE_DIR = "/model-cache"   # inside the Modal Volume

# Plain string GPU spec (modal.gpu.* objects are deprecated)
GPU_CONFIG = "A100-80GB"

# ---------------------------------------------------------------------------
# Persistent volume – weights are downloaded once and reused across cold starts
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("infinity-parser2-weights", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libgl1", "libglib2.0-0")
    # Torch (cu124 wheels are compatible with CUDA 12.8)
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Install build dependencies required by flash-attn
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    # FlashAttention-2 pre-built wheel
    .pip_install(
        "flash-attn==2.7.4.post1",
        extra_options="--no-build-isolation",
    )
    # Infinity-Parser2 (using transformers backend to avoid vLLM dependency conflicts)
    .pip_install("infinity_parser2", "accelerate")
    .pip_install("causal-conv1d", "flash-linear-attention")
    # Misc helpers
    .pip_install("huggingface_hub[cli]", "Pillow", "fastapi", "qwen_vl_utils", "pymupdf", "git+https://github.com/ozeliger/pyairports.git")
)

app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# Helper: download model weights into the Volume (run once)
# ---------------------------------------------------------------------------
@app.function(
    volumes={MODEL_CACHE_DIR: volume},
    gpu=None,           # CPU-only download
    timeout=3600,
)
def download_model():
    """Pull model weights from HuggingFace into the persistent Volume."""
    from huggingface_hub import snapshot_download

    local_dir = Path(MODEL_CACHE_DIR) / MODEL_ID.replace("/", "--")
    if (local_dir / "config.json").exists():
        print(f"Model already cached at {local_dir}")
        return str(local_dir)

    print(f"Downloading {MODEL_ID} -> {local_dir} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(local_dir),
        ignore_patterns=["*.pt", "original/*"],
    )
    volume.commit()
    print("Download complete.")
    return str(local_dir)


# ---------------------------------------------------------------------------
# Parser class - loaded once per container, reused across requests
# ---------------------------------------------------------------------------
@app.cls(
    gpu=GPU_CONFIG,
    volumes={MODEL_CACHE_DIR: volume},
    timeout=600,
    scaledown_window=300,
)
class Parser:
    @modal.enter()
    def load_model(self):
        import sys
        from unittest.mock import MagicMock
        sys.modules["vllm"] = MagicMock()

        from infinity_parser2 import InfinityParser2

        model_path = Path(MODEL_CACHE_DIR) / MODEL_ID.replace("/", "--")
        model_name = str(model_path) if model_path.exists() else MODEL_ID

        print(f"Loading model from: {model_name}")
        self.parser = InfinityParser2(
            model_name=model_name,
            backend="transformers",
            device="cuda",
        )
        print("Model ready.")

    @modal.method()
    def parse_bytes(
        self,
        file_bytes: bytes,
        filename: str = "input.pdf",
        task_type: str = "doc2json",
        output_format: str = "md",
    ) -> str:
        import tempfile

        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            result = self.parser.parse(
                tmp_path,
                task_type=task_type,
                output_format=output_format,
            )
        finally:
            os.unlink(tmp_path)

        return result

    @modal.method()
    def parse_base64(
        self,
        b64_content: str,
        filename: str = "input.pdf",
        task_type: str = "doc2json",
        output_format: str = "md",
    ) -> str:
        raw = base64.b64decode(b64_content)
        return self.parse_bytes.local(raw, filename, task_type, output_format)


# ---------------------------------------------------------------------------
# Web endpoint - deploy with `modal deploy`
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU_CONFIG,
    volumes={MODEL_CACHE_DIR: volume},
    timeout=300,
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST", docs=True)   # was: @modal.web_endpoint (deprecated)
def parse_endpoint(item: dict) -> dict:
    """
    REST endpoint. POST JSON body:
    {
      "content_b64":   "<base64-encoded file bytes>",
      "filename":      "my_report.pdf",   // optional
      "task_type":     "doc2json",        // optional
      "output_format": "md"              // optional
    }
    Returns: {"result": "<markdown or json string>"}
    """
    parser = Parser()
    result = parser.parse_base64.remote(
        b64_content=item["content_b64"],
        filename=item.get("filename", "input.pdf"),
        task_type=item.get("task_type", "doc2json"),
        output_format=item.get("output_format", "md"),
    )
    return {"result": result}


# ---------------------------------------------------------------------------
# Local CLI entrypoint:  modal run infinity_parser2_modal.py --input doc.pdf
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input: str = "",
    task_type: str = "doc2json",
    output_format: str = "md",
    download_only: bool = False,
):
    print("=== Ensuring model weights are cached ===")
    download_model.remote()

    if download_only:
        print("Done - weights cached. Exiting.")
        return

    if not input:
        print("Pass --input <path> to parse a file, or --download-only to cache weights.")
        return

    file_path = Path(input)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"=== Parsing {file_path.name} (task={task_type}, fmt={output_format}) ===")
    raw_bytes = file_path.read_bytes()

    parser = Parser()
    result = parser.parse_bytes.remote(
        file_bytes=raw_bytes,
        filename=file_path.name,
        task_type=task_type,
        output_format=output_format,
    )

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
