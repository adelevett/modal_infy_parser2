"""
Modal deployment for Infinity-Parser2-Pro
=========================================
Usage:
  modal run infinity_parser2_modal.py --input path/to/doc.pdf
  modal deploy infinity_parser2_modal.py

After deploying, use the submit+poll pattern for long-running jobs:
  1. POST /submit  → returns {"call_id": "..."}
  2. GET  /result/{call_id} → returns {"status": "pending"} or {"result": "..."}
"""

import os
import base64
from pathlib import Path

import modal

APP_NAME        = "infinity-parser2"
MODEL_ID        = "infly/Infinity-Parser2-Pro"
MODEL_CACHE_DIR = "/model-cache"
GPU_CONFIG      = "A100-80GB"

volume = modal.Volume.from_name("infinity-parser2-weights", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        # Use the README-recommended CUDA 12.8 compatible PyTorch builds.
        # Note: choose the exact wheel matching CUDA/toolchain in the target image.
        "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    # FlashAttention matching README recommendation. Keep --no-build-isolation
    # to avoid build isolation issues in some environments.
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    # Uninstall known conflicting packages at image build time so vLLM can
    # install cleanly. Avoid uninstalling `numpy` here to prevent breaking
    # other dependencies during build.
    .run_commands("pip uninstall -y pytorch-triton opencv-python opencv-python-headless || true")
    .pip_install("vllm==0.17.1")
    .pip_install("infinity_parser2", "accelerate")
    .pip_install("huggingface_hub[cli]", "Pillow", "fastapi", "qwen_vl_utils", "pymupdf")
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Weight download (run once)
# ---------------------------------------------------------------------------
@app.function(volumes={MODEL_CACHE_DIR: volume}, gpu=None, timeout=3600)
def download_model():
    from huggingface_hub import snapshot_download
    local_dir = Path(MODEL_CACHE_DIR) / MODEL_ID.replace("/", "--")
    if (local_dir / "config.json").exists():
        print(f"Already cached at {local_dir}")
        return str(local_dir)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(local_dir),
                      ignore_patterns=["*.pt", "original/*"])
    volume.commit()
    return str(local_dir)


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------
@app.cls(gpu=GPU_CONFIG, volumes={MODEL_CACHE_DIR: volume},
         timeout=600, scaledown_window=300)
class Parser:
    @modal.enter()
    def load_model(self):
        # At image-build time we uninstall conflicting packages; at runtime
        # prefer `vllm-engine` if available. Avoid performing pip
        # uninstall/install at cold-start to prevent long latency and
        # read-only filesystem errors in container layers.
        from infinity_parser2 import InfinityParser2
        model_path = Path(MODEL_CACHE_DIR) / MODEL_ID.replace("/", "--")
        model_name = str(model_path) if model_path.exists() else MODEL_ID
        print(f"Loading model from: {model_name}")

        # Prefer vllm-engine backend; fall back to transformers if vllm import fails.
        try:
            import vllm  # re-check after attempted install
            backend = "vllm-engine"
            parser_kwargs = {"backend": backend}
        except Exception:
            backend = "transformers"
            parser_kwargs = {"backend": backend, "device": "cuda"}

        self.parser = InfinityParser2(model_name=model_name, **parser_kwargs)
        print(f"Model ready. Backend={backend}")

    @modal.method()
    def parse_bytes(self, file_bytes: bytes, filename: str = "input.pdf",
                    task_type: str = "doc2json", output_format: str = "md") -> str:
        import tempfile
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return self.parser.parse(tmp_path, task_type=task_type, output_format=output_format)
        finally:
            os.unlink(tmp_path)

    @modal.method()
    def parse_base64(self, b64_content: str, filename: str = "input.pdf",
                     task_type: str = "doc2json", output_format: str = "md") -> str:
        # Decode base64 and parse directly in this method to avoid
        # ambiguity about whether `self.parse_bytes(...)` would be invoked
        # as a local call or a Modal-managed method. Inlining keeps the
        # behavior explicit and avoids surprising local/remote semantics.
        import tempfile
        data = base64.b64decode(b64_content)
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return self.parser.parse(tmp_path, task_type=task_type, output_format=output_format)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Web API — submit/poll pattern to avoid the 150s HTTP gateway timeout
# ---------------------------------------------------------------------------
@app.function(timeout=10)  # this endpoint just spawns and returns immediately
@modal.fastapi_endpoint(method="POST", docs=True)
def submit(item: dict) -> dict:
    """
    Submit a parse job. Returns immediately with a call_id.

    POST body:
      { "content_b64": "...", "filename": "doc.pdf",
        "task_type": "doc2json", "output_format": "json" }

    Returns:
      { "call_id": "fc-xxxx" }
    """
    parser = Parser()
    call = parser.parse_base64.spawn(
        b64_content=item["content_b64"],
        filename=item.get("filename", "input.pdf"),
        task_type=item.get("task_type", "doc2json"),
        output_format=item.get("output_format", "md"),
    )
    return {"call_id": call.object_id}


@app.function(timeout=10)  # just a lookup, also fast
@modal.fastapi_endpoint(method="GET", docs=True)
def result(call_id: str) -> dict:
    """
    Poll for results. GET /result?call_id=fc-xxxx

    Returns:
      { "status": "pending" }          — still running
      { "status": "complete", "result": "..." }  — done
    """
    from modal import FunctionCall
    fc = FunctionCall.from_id(call_id)
    try:
        output = fc.get(timeout=0)   # timeout=0 = poll without waiting
        return {"status": "complete", "result": output}
    except TimeoutError:
        return {"status": "pending"}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(input: str = "", task_type: str = "doc2json",
         output_format: str = "md", download_only: bool = False):
    import time

    print("=== Ensuring model weights are cached ===")
    download_model.remote()

    if download_only:
        print("Done."); return
    if not input:
        print("Pass --input <path>"); return

    file_path = Path(input)
    if not file_path.exists():
        raise FileNotFoundError(f"Not found: {file_path}")

    print(f"=== Parsing {file_path.name} (task={task_type}, fmt={output_format}) ===")
    raw_bytes = file_path.read_bytes()

    parser = Parser()
    result_str = parser.parse_bytes.remote(
        file_bytes=raw_bytes, filename=file_path.name,
        task_type=task_type, output_format=output_format,
    )
    print("\n" + "=" * 60)
    print(result_str)
    print("=" * 60)