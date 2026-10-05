import re
import time
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

APP_TITLE = "🏕️ National Park Camping Guide"
DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"
TOP_K = 5
MIN_SIMILARITY = 0.34
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150

SYSTEM_PROMPT = """คุณคือ National Park Camping Guide ผู้ช่วยด้านการเดินป่าและกางเต็นท์
ในอุทยานแห่งชาติของประเทศไทย

กฎการตอบ:
1. ใช้เฉพาะข้อมูลที่อยู่ใน CONTEXT เท่านั้น
2. ห้ามเดา ห้ามเติมข้อมูลจากความรู้ทั่วไป และห้ามสร้างตัวเลข ราคา เบอร์โทรศัพท์
   หรือข้อกำหนดที่ไม่มีใน CONTEXT
3. หาก CONTEXT ไม่มีข้อมูลเพียงพอที่จะตอบ ให้ตอบว่า
   "ไม่พบข้อมูลในเอกสาร"
4. ตอบให้กระชับ เข้าใจง่าย และตอบเป็นภาษาเดียวกับคำถามเมื่อทำได้
5. หากมีข้อมูลหลายเอกสาร ให้สังเคราะห์เฉพาะข้อมูลที่ไม่ขัดแย้งกัน
6. ห้ามอ้างว่าได้ตรวจสอบเว็บไซต์แบบเรียลไทม์ เพราะระบบนี้ตอบจากคลังเอกสารเท่านั้น
7. ไม่ต้องสร้างหัวข้อ "แหล่งอ้างอิง" เอง เพราะแอปจะแสดง Source แยกให้

CONTEXT:
{context}

QUESTION:
{question}
"""


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{paragraph}".strip()

            # If one paragraph is unusually long, split it by sentence/character.
            while len(current) > chunk_size:
                cut = current[:chunk_size]
                split_at = max(cut.rfind("。"), cut.rfind("."), cut.rfind("\n"), cut.rfind(" "))
                if split_at < chunk_size * 0.55:
                    split_at = chunk_size
                chunks.append(current[:split_at].strip())
                current = current[max(0, split_at - overlap):].strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= 80]


@st.cache_resource(show_spinner="กำลังโหลด Embedding Model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="กำลังสร้าง Vector Index...")
def build_index():
    if not DATA_DIR.exists():
        raise FileNotFoundError("ไม่พบโฟลเดอร์ data/")

    documents = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        cleaned = clean_text(text)
        for idx, chunk in enumerate(chunk_text(cleaned)):
            documents.append(
                {
                    "source": path.name,
                    "chunk_id": idx,
                    "text": chunk,
                }
            )

    if not documents:
        raise ValueError("ไม่พบเอกสาร .txt ใน data/")

    model = load_embedding_model()
    texts = [d["text"] for d in documents]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, documents


def retrieve(query: str, index, documents, model, top_k: int = TOP_K):
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(query_vector, min(top_k, len(documents)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append(
            {
                "score": float(score),
                "source": documents[idx]["source"],
                "chunk_id": documents[idx]["chunk_id"],
                "text": documents[idx]["text"],
            }
        )
    return results


def make_context(results: List[Dict]) -> str:
    blocks = []
    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[Context {i} | Source: {item['source']} | Chunk: {item['chunk_id']}]\n"
            f"{item['text']}"
        )
    return "\n\n".join(blocks)

def get_gemini_client():
    if "GEMINI_API_KEY" not in st.secrets:
        return None

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


def generate_answer(question: str, context: str, history: List[Dict]) -> str:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน Streamlit Secrets")

    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    # Try primary model with exponential backoff, then fall back to a secondary model.
    models_to_try = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]
    max_retries = 3

    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=900,
                    ),
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                is_503 = "503" in error_str or "UNAVAILABLE" in error_str
                is_last_attempt = attempt == max_retries - 1

                if is_503 and not is_last_attempt:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait)
                    continue  # retry same model
                elif is_503 and is_last_attempt:
                    break  # try next model
                else:
                    raise  # non-503 error — raise immediately

    raise RuntimeError(
        f"Gemini API ไม่พร้อมใช้งานชั่วคราว (503) ลองอีกครั้งในอีกสักครู่"
    )




GREETING_PATTERNS = re.compile(
    r"^\s*(hi+|hello+|hey+|สวัสดี|หวัดดี|ดีจ้า|ดีครับ|ดีค่ะ|yo+|aloha|howdy|"
    r"ขอบคุณ|thank|thanks|bye|goodbye|ลาก่อน|แล้วเจอกัน|เย่|ok|โอเค)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

GREETING_REPLY = (
    "สวัสดีครับ! 🏕️ ผมคือ **National Park Camping Guide**\n\n"
    "ผมช่วยตอบคำถามเกี่ยวกับ:\n"
    "- 🏕️ การกางเต็นท์และแคมป์ปิ้ง\n"
    "- 🥾 การเดินป่าและความปลอดภัย\n"
    "- 📋 กฎระเบียบของอุทยานแห่งชาติ\n"
    "- 🏥 การรับมือเหตุฉุกเฉิน\n"
    "- 💰 ค่าธรรมเนียมและสิ่งอำนวยความสะดวก\n\n"
    "ลองถามได้เลยครับ! 😊"
)

NOT_FOUND_REPLY = (
    "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร 📄\n\n"
    "ระบบนี้ตอบเฉพาะเรื่อง **อุทยานแห่งชาติ** เช่น กฎระเบียบ การกางเต็นท์ "
    "การเดินป่า ค่าธรรมเนียม และเหตุฉุกเฉิน\n\n"
    "💡 ลองดูตัวอย่างคำถามในแถบด้านซ้ายได้ครับ"
)


def is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERNS.match(text.strip()))


def unique_sources(results: List[Dict]) -> List[str]:
    seen = set()
    output = []
    for item in results:
        if item["source"] not in seen:
            seen.add(item["source"])
            output.append(item["source"])
    return output


def main():
    st.set_page_config(
        page_title="National Park Camping Guide",
        page_icon="🏕️",
        layout="wide",
    )

    st.title(APP_TITLE)
    st.caption("ผู้ช่วยตอบคำถามจากคลังเอกสารความรู้ด้วย Retrieval-Augmented Generation (RAG)")

    with st.sidebar:
        st.header("ℹ️ เกี่ยวกับระบบ")
        st.write(
            "ระบบจะค้นหาเนื้อหาที่เกี่ยวข้องจากเอกสารใน `data/` "
            "ก่อนส่ง Context ให้ LLM สร้างคำตอบ"
        )
        st.divider()
        st.subheader("ตัวอย่างคำถาม")
        suggestions = [
            "ต้องเตรียมอุปกรณ์อะไรสำหรับการกางเต็นท์?",
            "มีกฎอะไรเกี่ยวกับการจัดการขยะ?",
            "มีสิ่งอำนวยความสะดวกอะไรบ้าง?",
            "หากเกิดเหตุฉุกเฉินควรทำอย่างไร?",
        ]
        for suggestion in suggestions:
            if st.button(suggestion, use_container_width=True):
                st.session_state["pending_question"] = suggestion

        st.divider()
        st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
        st.caption(f"LLM: `{GEMINI_MODEL}`")

    try:
        index, documents = build_index()
        model = load_embedding_model()
    except Exception as exc:
        st.error(f"ไม่สามารถเตรียม RAG Index ได้: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = {}

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and i in st.session_state.sources:
                sources = st.session_state.sources[i]
                if sources:
                    with st.expander("📚 แหล่งข้อมูล"):
                        for source in sources:
                            st.write(f"- `{source}`")

    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("ถามเกี่ยวกับการเดินป่า การกางเต็นท์ หรือกฎของอุทยาน...")

    if pending and not question:
        question = pending

    if question:
        question = question.strip()
        if not question:
            st.warning("กรุณาพิมพ์คำถาม")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # --- Greeting / small-talk fast-path ---
            if is_greeting(question):
                answer = GREETING_REPLY
                sources = []
            else:
                with st.status("🔎 กำลังค้นหาข้อมูลจากเอกสาร...", expanded=False):
                    results = retrieve(question, index, documents, model, TOP_K)

                best_score = results[0]["score"] if results else 0.0
                useful = [r for r in results if r["score"] >= MIN_SIMILARITY]

                if not useful or best_score < MIN_SIMILARITY:
                    answer = NOT_FOUND_REPLY
                    sources = []
                else:
                    context = make_context(useful)
                    try:
                        with st.status("🤖 กำลังสร้างคำตอบจาก Context...", expanded=False):
                            answer = generate_answer(
                                question,
                                context,
                                st.session_state.messages[:-1],
                            )
                    except Exception as exc:
                        st.error(f"เกิดข้อผิดพลาดในการเรียก LLM: {exc}")
                        st.stop()
                    sources = unique_sources(useful)

            st.markdown(answer)

            if sources:
                with st.expander("📚 แหล่งข้อมูลที่ใช้ตอบ"):
                    for source in sources:
                        st.write(f"- `{source}`")

        assistant_index = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.sources[assistant_index] = sources
        st.rerun()


if __name__ == "__main__":
    main()
