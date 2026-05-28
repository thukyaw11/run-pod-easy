import base64
import io
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import runpod
import requests

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (pymupdf) is required.") from e

try:
    import easyocr
except Exception as e:
    raise RuntimeError("easyocr is required.") from e

try:
    import numpy as np
    from PIL import Image
except Exception as e:
    raise RuntimeError("Pillow and numpy are required.") from e

try:
    import pytesseract
    from pytesseract import Output as TessOutput
except Exception as e:
    raise RuntimeError("pytesseract is required.") from e


# Global cache — avoids reloading model weights on every request
_READER_CACHE: Dict[Tuple[Tuple[str, ...], bool, bool], easyocr.Reader] = {}


def get_reader(languages: List[str], use_gpu: bool, cudnn_benchmark: bool = False) -> easyocr.Reader:
    key = (tuple(languages), bool(use_gpu), bool(cudnn_benchmark))
    if key not in _READER_CACHE:
        _READER_CACHE[key] = easyocr.Reader(languages, gpu=use_gpu, cudnn_benchmark=cudnn_benchmark)
    return _READER_CACHE[key]


def _fetch_pdf(job_input: dict) -> bytes:
    if "file_url" in job_input:
        with urllib.request.urlopen(job_input["file_url"]) as r:
            return r.read()
    if "file_base64" in job_input:
        return base64.b64decode(job_input["file_base64"])
    # backward compat
    if "pdf_url" in job_input:
        resp = requests.get(job_input["pdf_url"], timeout=60)
        resp.raise_for_status()
        return resp.content
    raise ValueError("Input must have 'file_url', 'file_base64', or 'pdf_url'.")


def pdf_pages_to_png_bytes(pdf_bytes: bytes, dpi: int = 200) -> List[Tuple[int, bytes, Tuple[int, int]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append((page_index, pix.tobytes("png"), (pix.width, pix.height)))
    return images


def correct_orientation_bytes(image_bytes: bytes) -> Tuple[bytes, Tuple[int, int], Dict[str, Any]]:
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB")
        osd: Dict[str, Any] = {}
        rotate_degrees = 0
        try:
            osd = pytesseract.image_to_osd(rgb, output_type=TessOutput.DICT)
            rotate_degrees = int(osd.get("rotate", 0) or 0)
        except Exception:
            osd = {"rotate": 0}

        rotated = rgb
        if rotate_degrees % 360 != 0:
            rotated = rgb.rotate(0 - rotate_degrees, expand=True)

        buf = io.BytesIO()
        rotated.save(buf, format="PNG")
        data = buf.getvalue()
        w, h = rotated.size
        return data, (w, h), osd


def _build_page_text(results: List[Any]) -> str:
    """Sort EasyOCR results top-to-bottom, left-to-right and join as plain text."""
    sorted_results = sorted(
        results,
        key=lambda r: (
            min(pt[1] for pt in r[0]),  # top-most y
            min(pt[0] for pt in r[0]),  # left-most x
        ),
    )
    return "\n".join(r[1] for r in sorted_results if r[1].strip())


def handler(job):
    """
    action = "ocr" (default):
      Input:  { "file_url": "...", "languages": ["ch_sim","en"], "gpu": true, "dpi": 200 }
      Yields: { "page": 1, "total_pages": 5, "text": "..." }  per page

    action = "detect_orientation":
      Input:  { "action": "detect_orientation", "file_url": "..." }
      Yields: { "total_pages": 5, "landscape_pages": [2, 4] }
    """
    inp = job.get("input", {})
    action = inp.get("action", "ocr")

    try:
        pdf_bytes = _fetch_pdf(inp)
    except Exception as exc:
        yield {"error": f"Failed to load file: {exc}"}
        return

    dpi = int(inp.get("dpi", 200))
    try:
        page_images = pdf_pages_to_png_bytes(pdf_bytes, dpi=dpi)
    except Exception as exc:
        yield {"error": f"Failed to render PDF: {exc}"}
        return

    total_pages = len(page_images)

    if action == "detect_orientation":
        landscape_pages = []
        for idx, img_bytes, _ in page_images:
            try:
                _, _, osd = correct_orientation_bytes(img_bytes)
                rotate = int(osd.get("rotate", 0) or 0)
                if rotate in (90, 270):
                    landscape_pages.append(idx + 1)  # 1-indexed
            except Exception:
                pass
        yield {"total_pages": total_pages, "landscape_pages": landscape_pages}
        return

    # action == "ocr"
    default_langs = os.getenv("READER_LANGS", "ch_sim,en").split(",")
    languages = inp.get("languages") or [l.strip() for l in default_langs if l.strip()]
    use_gpu = bool(inp.get("gpu", True))
    cudnn_benchmark = bool(inp.get("cudnn_benchmark", False))

    try:
        reader = get_reader(languages, use_gpu, cudnn_benchmark=cudnn_benchmark)
    except Exception as exc:
        yield {"error": f"Failed to init reader: {exc}"}
        return

    for idx, img_bytes, _ in page_images:
        page_no = idx + 1
        try:
            print(f"[easyocr] OCR page {page_no}/{total_pages}")
            corrected_bytes, _, _ = correct_orientation_bytes(img_bytes)
            results = reader.readtext(corrected_bytes, detail=1)
            text = _build_page_text(results)
            yield {"page": page_no, "total_pages": total_pages, "text": text}
        except Exception as exc:
            yield {"error": f"Page {page_no} failed: {exc}"}
            return


runpod.serverless.start({"handler": handler})
