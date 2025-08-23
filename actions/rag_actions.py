#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG con Milvus **Lite** (embebido) + **Ollama** para generación.
- Ingesta un PDF a una colección local (milvus_lite.db)
- /chat: recupera contextos desde Milvus Lite y llama a Ollama para responder

Uso:
  python rag_milvus_lite_pdf_ollama.py --pdf_path ./mi_doc.pdf --recreate
  uvicorn rag_milvus_lite_pdf_ollama:app --host 0.0.0.0 --port 8000

ENV:
  OLLAMA_BASE_URL (default: http://localhost:11434)
  OLLAMA_MODEL    (default: phi3:mini)  # ligero
  MILVUS_LITE_PATH (default: ./milvus_lite.db)
"""
import argparse
import os
import re
from typing import List, Optional, Dict, Any
import time
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

#import socketio






def clean_text(t: str) -> str:
    # Une palabras cortadas por guion al final de línea, pero preserva saltos
    t = re.sub(r'-\s*\n', '', t)              # "infor-\nmación" -> "información"
    # Normaliza espacios alrededor de saltos y colapsa múltiples saltos en uno
    t = re.sub(r'[ \t]*\n+[ \t]*', '\n', t)   # limpia márgenes de \n
    # Opcional: deja doble salto entre párrafos largos
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

# ---------------- Config ----------------
COLLECTION_NAME = "bago_pdf"
EMB_DIM = 384  # all-MiniLM-L6-v2

# Milvus Lite solo soporta FLAT (los demás se ignoran)
INDEX_PARAMS = {"index_type": "FLAT", "metric_type": "COSINE"}  # sin "params"
SEARCH_PARAMS = {"metric_type": "COSINE", "params": {}}         # sin ef/nprobe

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Modelo ligero por defecto para equipos con poca RAM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
MILVUS_LITE_PATH = os.getenv("MILVUS_LITE_PATH", "data/milvus/milvus_lite.db")

import re

def clean_text(t: str) -> str:
    # une palabras cortadas por guion al final de línea, colapsa saltos
    t = re.sub(r'-\n', '', t)
    t = re.sub(r'\s*\n+\s*', ' ', t)
    return t



# ---------------- Utilidades PDF/Chunks ----------------
def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def split_sentences(text: str) -> List[str]:
    # División simple por signos de puntuación seguidos de espacio/salto
    # Evita dependencias pesadas; ajusta si tu PDF tiene muchos puntos abreviados.
    parts = re.split(r'(?<=[\.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(
    text: str,
    max_len: int = 1000,     # tamaño objetivo seguro (< 4096 de Milvus)
    overlap: int = 150,
    hard_cap: int = 3800     # nunca exceder esto (por debajo de 4096)
) -> List[str]:
    # Trabaja por párrafos primero (manteniendo \n)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    def push_current():
        nonlocal current
        if current:
            # corte duro final si se pasó por cualquier motivo
            for i in range(0, len(current), hard_cap):
                chunks.append(current[i:i+hard_cap])
            current = ""

    for para in paragraphs:
        # si el párrafo ya es inmenso, lo partimos por oraciones
        if len(para) > max_len:
            sents = split_sentences(para)
            for s in sents:
                if len(s) > hard_cap:
                    # última barrera: troceo por longitud pura
                    for i in range(0, len(s), hard_cap):
                        piece = s[i:i+hard_cap]
                        if not current:
                            current = piece
                        elif len(current) + 1 + len(piece) <= max_len:
                            current = (current + " " + piece).strip()
                        else:
                            push_current()
                            # añade solapamiento
                            tail = current[-overlap:] if current else ""
                            current = (tail + " " + piece).strip()
                else:
                    if not current:
                        current = s
                    elif len(current) + 1 + len(s) <= max_len:
                        current = (current + " " + s).strip()
                    else:
                        push_current()
                        tail = current[-overlap:] if current else ""
                        current = (tail + " " + s).strip()
            push_current()
        else:
            # párrafo corto: empaquetar con el actual
            if not current:
                current = para
            elif len(current) + 1 + len(para) <= max_len:
                current = (current + " " + para).strip()
            else:
                push_current()
                tail = current[-overlap:] if current else ""
                current = (tail + " " + para).strip()

    if current:
        push_current()

    # salvaguarda final
    safe = []
    for c in chunks:
        if len(c) <= hard_cap:
            safe.append(c)
        else:
            for i in range(0, len(c), hard_cap):
                safe.append(c[i:i+hard_cap])
    return safe


# ---------------- Milvus Lite helpers ----------------
def connect_milvus_lite():
    db_path = os.getenv("MILVUS_LITE_PATH", "./milvus_lite.db")
    db_path = os.path.abspath(db_path)          # path absoluto
    parent = os.path.dirname(db_path) or "."
    os.makedirs(parent, exist_ok=True)          # crea carpeta si falta
    print(f"[Milvus Lite] usando archivo: {db_path}")
    connections.connect(alias="default", uri="data/milvus/milvus_lite.db")


def ensure_collection(recreate: bool = False) -> Collection:
    if recreate and utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),  # puedes subir a 8192 si prefieres
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM),
        ]
        schema = CollectionSchema(fields, description="Document chunks from PDF")
        col = Collection(COLLECTION_NAME, schema)
        col.create_index(field_name="embedding", index_params=INDEX_PARAMS)
    else:
        col = Collection(COLLECTION_NAME)
    col.load()
    return col


def ingest_pdf(pdf_path: str, model_name: str = "all-MiniLM-L6-v2", recreate: bool = False) -> int:
    raw = read_pdf_text(pdf_path)
    text = clean_text(raw)
    chunks = chunk_text(text, max_len=1000, overlap=150, hard_cap=3800)
    if not chunks:
        raise RuntimeError("No se pudo extraer texto del PDF. ¿Está escaneado? Usa OCR (pytesseract/ocrmypdf).")
    if any(len(c) > 4096 for c in chunks):
        raise RuntimeError("Se generaron chunks > 4096; ajusta max_len/hard_cap.")

    print(f"[Ingesta] {len(chunks)} chunks")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True).tolist()

    col = ensure_collection(recreate=recreate)
    mr = col.insert(data=[chunks, embeddings], fields=["text", "embedding"])
    col.flush()
    print(f"[Ingesta] Insertados {mr.insert_count} vectores en {COLLECTION_NAME}")
    return mr.insert_count

# ---------------- Ollama helpers ----------------
def call_ollama_chat(model: str, system: str, user: str, options: Optional[Dict[str, Any]] = None) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        # Mantén el modelo caliente para que no lo descargue cada vez
        "keep_alive": "24h",
    }
    if options:
        payload["options"] = options

    # Reintentos: 3 intentos con backoff exponencial
    last_err = None
    for attempt in range(3):
        try:
            # Subir timeout a 600 s para primeras cargas
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ReadTimeout as e:
            last_err = e
        except requests.RequestException as e:
            last_err = e
        time.sleep(2 ** attempt)  # 1s, 2s, 4s

    raise RuntimeError(
        f"No pude obtener respuesta de Ollama en {url} usando el modelo '{model}'. "
        f"¿Está corriendo y el modelo está descargado? Detalle: {last_err}"
    )

def build_system_prompt() -> str:
    """return (
        "Eres un asistente que responde en español usando exclusivamente el CONTEXTO proporcionado. "
        "Si la respuesta no está en el contexto, responde con: 'No se encuentra en el documento.' "
        "Cuando sea útil, cita brevemente el fragmento relevante entre comillas."
    )"""
    return (
     """Rol: Eres un asistente especializado en vademécum médico.

        Objetivo: Tu función es brindar información exacta, clara y concisa sobre medicamentos únicamente a partir de datos oficiales del vademécum.

        Reglas de seguridad y alcance:
        1. Limítate a información factual: nombre comercial, principio activo, dosis genéricas, contraindicaciones, efectos secundarios, presentaciones, interacciones y advertencias.
        2. No inventes información. Si no tienes datos de un medicamento, responde: "No tengo información registrada sobre ese medicamento."
        3. No des diagnósticos médicos ni recomendaciones personalizadas de tratamiento.
        4. Si el usuario pide dosis personalizadas, preparación casera de fármacos, mezclas no documentadas o usos fuera del vademécum, rechaza educadamente con: "No puedo dar indicaciones personalizadas. Solo puedo compartir la información registrada en el vademécum. Consulte a un médico o farmacéutico."
        5. Si detectas una situación de emergencia (ejemplo: sobredosis, intoxicación, malestar grave), responde de inmediato: "Esto puede ser una emergencia. Acuda de inmediato a un centro de salud o llame a los servicios de emergencia."
        6. Siempre incluye al final de cada respuesta la advertencia: "⚠️ Esta información es solo de carácter informativo y no sustituye la consulta médica profesional."

        Estilo de respuesta: Usa un lenguaje claro, directo y estructurado. Cuando sea posible, organiza la información en listas para mayor legibilidad."""
            )

def build_user_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    return (
        f"CONTEXTO:\n{ctx}\n\n"
        f"---\n"
        f"Pregunta: {question}\n"
        f"Responde de forma concisa y fiel al contexto."
    )

# ---------------- API ----------------
app = FastAPI(title="RAG Milvus Lite + Ollama")
'''sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')


# Monta el servidor Socket.IO en FastAPI
app_socketio = socketio.ASGIApp(sio, other_asgi_app=app)

# Ejemplo de evento
@sio.event
async def connect(sid, environ):
    print(f"Cliente conectado: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Cliente desconectado: {sid}")

@sio.event
async def chat_message(sid, data):
    print(f"Mensaje recibido: {data}")
    await sio.emit('chat_response', {'response': 'Mensaje recibido'}, to=sid)

# Para correr: uvicorn rag_actions:app_socketio --host 0.0.0.0 --port 8000

'''

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.1
    num_ctx: int = 2048  # por defecto más bajo para reducir RAM

@app.post("/chat")
def chat(q: Query):
    if not connections.has_connection("default"):
        connect_milvus_lite()
    if not utility.has_collection(COLLECTION_NAME):
        return {"error": f"No existe la colección {COLLECTION_NAME}. Ingresa primero un PDF."}

    col = Collection(COLLECTION_NAME)
    col.load()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qvec = embedder.encode([q.question]).tolist()

    results = col.search(
        qvec,
        anns_field="embedding",
        param=SEARCH_PARAMS,
        output_fields=["text"],
        limit=q.top_k
        )

    first = results[0]

# Normaliza a lista (soporta Hits, SequenceIterator, etc.)
    try:
        hits_list = list(first)
    except TypeError:
        hits_list = first  # por si ya es list-like

    contexts = []
    distances = []
    for h in hits_list:
        # En pymilvus 2.4.x: h.entity.get("text") y h.distance
        txt = ""
        try:
            ent = getattr(h, "entity", None)
            if ent is not None and hasattr(ent, "get"):
                txt = ent.get("text") or ""
        except Exception:
            pass
        contexts.append(txt)
        distances.append(float(getattr(h, "distance", 0.0)))
        
    system = build_system_prompt()
    user = build_user_prompt(q.question, contexts)
    try:
        answer = call_ollama_chat(
            model=OLLAMA_MODEL,
            system=system,
            user=user,
            options={"temperature": q.temperature, "num_ctx": q.num_ctx}
        )
    except Exception as e:
        return {
            "error": f"Fallo al llamar a Ollama ({OLLAMA_BASE_URL}). ¿Está corriendo y tienes el modelo '{OLLAMA_MODEL}'?",
            "detail": str(e),
            "contexts_preview": contexts[:2]
        }

    return {
        "question": q.question,
        "answer": answer,
        "contexts_used": contexts,
        "distances": distances,
        "model": OLLAMA_MODEL
    }

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, required=True, help="Ruta al PDF a indexar")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Modelo de embeddings")
    parser.add_argument("--recreate", action="store_true", help="Recrear colección desde cero")
    args = parser.parse_args()

    connect_milvus_lite()
    ingest_pdf(args.pdf_path, model_name=args.model, recreate=args.recreate)
    print("Listo. Inicia la API con:  uvicorn rag_milvus_lite_pdf_ollama:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
