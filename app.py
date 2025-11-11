import os

import uuid
import time
import asyncio
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional

import fitz  # PyMuPDF
import numpy as np
import cv2
import pytesseract

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MAX_WORKERS  = int(os.getenv("MAX_WORKERS", "4"))
PAGE_TIMEOUT = int(os.getenv("PAGE_TIMEOUT", "25"))
RENDER_DPI   = int(os.getenv("RENDER_DPI", "300"))
MAX_FILE_MB  = int(os.getenv("MAX_FILE_MB", "100"))

app = FastAPI(title="PDF OCR API", version="1.2.0")

executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
JOBS: Dict[str, Dict[str, Any]] = {}  # job_id -> {status, progress, result, error}

# -----------------------------
# OCR utils
# -----------------------------
def pick_tesseract_lang() -> str:
    try:
        langs = pytesseract.get_languages(config='')
        if 'por' in langs and 'eng' in langs:
            return 'por+eng'
        if 'por' in langs:
            return 'por'
        return 'eng'
    except Exception:
        return 'eng'

TESS_LANG   = pick_tesseract_lang()
BASE_CONFIG = f'--oem 3 --psm 6 -l {TESS_LANG}'

def preprocess_image(img: np.ndarray) -> np.ndarray:
    # grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    # denoise
    gray = cv2.medianBlur(gray, 3)
    # equalize (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Otsu
    try:
        _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        th = eq
    # deskew simples
    th_inv = 255 - th
    coords = np.column_stack(np.where(th_inv > 0))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = th.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        th = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # adaptativa (fallback)
    try:
        adapt = cv2.adaptiveThreshold(
            th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
        )
        return adapt
    except Exception:
        return th

def render_page_to_image(page: fitz.Page, dpi: int = RENDER_DPI) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img

def ocr_page(img_rgb: np.ndarray, timeout: int = PAGE_TIMEOUT) -> str:
    processed = preprocess_image(img_rgb)
    try:
        text = pytesseract.image_to_string(processed, config=BASE_CONFIG, timeout=timeout)
        if not text or not text.strip():
            text = pytesseract.image_to_string(
                processed, config=f'--oem 3 --psm 4 -l {TESS_LANG}', timeout=timeout
            )
    except Exception as e:
        logging.warning(f"Tesseract falhou/timeout: {e}; fallback PSM 4.")
        text = pytesseract.image_to_string(
            processed, config=f'--oem 3 --psm 4 -l {TESS_LANG}', timeout=min(timeout, 20)
        )
    return (text or "").strip()

def extract_pdf_bytes(pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
    start = time.time()
    pages_out: List[Dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            # 1) texto embutido
            embedded = (page.get_text("text") or "").strip()
            if len(embedded) > 20:
                pages_out.append({"page": i + 1, "method": "text_layer", "text": embedded})
                continue
            # 2) OCR
            img = render_page_to_image(page, dpi=RENDER_DPI)
            text = ocr_page(img, timeout=PAGE_TIMEOUT)
            pages_out.append({"page": i + 1, "method": "ocr", "text": text})
    combined = "\n\n".join(
        f"--- Página {p['page']} ({p['method']}) ---\n{p['text']}" for p in pages_out
    ).strip()
    return {
        "filename": filename,
        "meta": {
            "pages": len(pages_out),
            "elapsed_sec": round(time.time() - start, 2),
            "dpi": RENDER_DPI,
            "tesseract_lang": TESS_LANG
        },
        "pages": pages_out,
        "combined_text": combined
    }

# -----------------------------
# helpers para aceitar QUALQUER nome de campo
# -----------------------------
async def collect_uploads_from_request(
    request: Request,
    file_param: Optional[UploadFile],
    files_param: Optional[List[UploadFile]],
) -> List[UploadFile]:
    uploads: List[UploadFile] = []
    # preferir os parâmetros mapeados
    if file_param is not None:
        uploads.append(file_param)
    if files_param:
        uploads.extend([u for u in files_param if u is not None])
    if uploads:
        return uploads
    # varrer todo o multipart
    try:
        form = await request.form()
        for _, value in form.multi_items():
            if isinstance(value, UploadFile):
                uploads.append(value)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, UploadFile):
                        uploads.append(v)
    except Exception as e:
        logging.warning(f"Falha lendo form: {e}")
    return uploads

# -----------------------------
# Models
# -----------------------------
class AsyncResponse(BaseModel):
    job_id: str
    accepted: int

class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "lang": TESS_LANG, "dpi": RENDER_DPI}

@app.post("/extract-sync")
async def extract_sync(
    request: Request,
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    uploads = await collect_uploads_from_request(request, file, files)
    if not uploads:
        raise HTTPException(status_code=422, detail="Nenhum PDF encontrado no multipart.")
    if len(uploads) > 1:
        logging.info(f"/extract-sync recebeu {len(uploads)} arquivos; processando apenas o primeiro.")
    uf = uploads[0]
    if not uf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, f"Arquivo inválido (não PDF): {uf.filename}")
    pdf_bytes = await uf.read()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, extract_pdf_bytes, pdf_bytes, uf.filename)
    return JSONResponse(result)

@app.post("/extract-async", status_code=202)
async def extract_async(
    request: Request,
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    uploads = await collect_uploads_from_request(request, file, files)
    if not uploads:
        raise HTTPException(status_code=422, detail="Nenhum PDF encontrado no multipart.")
    for uf in uploads:
        if not uf.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"Arquivo inválido (não PDF): {uf.filename}")

    # 1) BUFFER: leia os bytes AQUI, antes de retornar a resposta
    buffered: List[tuple[str, bytes]] = []
    for uf in uploads:
        data = await uf.read()
        if not data:
            raise HTTPException(400, f"Arquivo vazio: {uf.filename}")
        buffered.append((uf.filename, data))

    # 2) cria o job
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}

    # 3) worker usando os BYTES (não mais UploadFile)
    async def _worker(job_id: str, files_data: List[tuple[str, bytes]]):
        try:
            JOBS[job_id]["status"] = "running"
            all_results: List[Dict[str, Any]] = []
            total = len(files_data)

            loop = asyncio.get_running_loop()
            for idx, (fname, data) in enumerate(files_data, start=1):
                res = await loop.run_in_executor(executor, extract_pdf_bytes, data, fname)
                all_results.append(res)
                JOBS[job_id]["progress"] = idx / total

            JOBS[job_id]["result"] = {"files": all_results}
            JOBS[job_id]["status"] = "done"
        except Exception as e:
            logging.exception("Falha no job")
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["status"] = "error"

    asyncio.create_task(_worker(job_id, buffered))

    # 4) 202 + header com o job id (útil pro n8n)
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "accepted": len(buffered)},
        headers={"X-Job-Id": job_id, "Cache-Control": "no-store"},
        media_type="application/json"
    )

@app.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result": job.get("result"),
        "error": job.get("error"),
    }
