# 🏕️ National Park Camping Guide

Web Application Chatbot สำหรับตอบคำถามเกี่ยวกับการเดินป่าและกางเต็นท์ในอุทยานแห่งชาติ โดยใช้ Retrieval-Augmented Generation (RAG)

## จุดเด่น

- Document Loading + Text Cleaning
- Chunking พร้อม overlap
- Sentence Embedding สำหรับไทย/อังกฤษ
- FAISS Vector Search
- Prompt ที่บังคับให้ตอบจาก Context เท่านั้น
- Groq LLM
- แสดง Source ของคำตอบ
- ปฏิเสธเมื่อไม่พบข้อมูล
- Streamlit Chat Interface
- Cache Embedding Model และ FAISS Index

## Architecture

```text
data/*.txt
   ↓
Cleaning
   ↓
Chunking
   ↓
Multilingual Sentence Embedding
   ↓
FAISS
   ↓
User Question
   ↓
Query Embedding
   ↓
Top-K Retrieval
   ↓
Context
   ↓
RAG Prompt
   ↓
Groq LLM
   ↓
Answer + Sources
```

## Technology

- Python
- Streamlit
- Sentence Transformers
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- FAISS
- Groq API
- `openai/gpt-oss-20b`

โมเดล Embedding นี้รองรับ 50 ภาษา จึงเหมาะกับชุดข้อมูลไทย/อังกฤษของโครงการ

## แหล่งข้อมูล

หัวข้อของโครงการอ้างอิงแนวทางและข้อมูลสาธารณะจาก **กรมอุทยานแห่งชาติ สัตว์ป่า และพันธุ์พืช (DNP)**

> ชุด `data/` ที่แนบมากับ repository นี้เป็น **ตัวอย่างคลังความรู้สำหรับพัฒนาและทดสอบ RAG** เพื่อให้โครงการรันได้ทันที นักศึกษาควรตรวจสอบและแทนที่/เพิ่มเติมด้วยเอกสารต้นฉบับจาก DNP ก่อนส่งงานจริง โดยเฉพาะข้อมูลที่เปลี่ยนแปลงได้ เช่น อัตราค่าบริการ เบอร์ติดต่อ และสถานะการเปิดพื้นที่

## โครงสร้าง

```text
national-park-rag/
├── app.py
├── requirements.txt
├── README.md
├── test_questions.csv
├── .gitignore
├── .streamlit/
│   └── secrets.toml        # local only, DO NOT commit
└── data/
    ├── 01_overview.txt
    ├── 02_park_rules.txt
    ├── 03_camping_rules.txt
    ├── 04_equipment.txt
    ├── 05_hiking_safety.txt
    ├── 06_facilities.txt
    ├── 07_fees.txt
    ├── 08_emergency.txt
    ├── 09_environment.txt
    └── 10_trip_planning.txt
```

## ติดตั้งในเครื่อง

แนะนำ Python 3.12 ให้ตรงกับสภาพแวดล้อมของ Streamlit Community Cloud

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd national-park-rag

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

ติดตั้ง:

```bash
pip install -r requirements.txt
```

## ตั้งค่า Groq API Key

สร้างไฟล์:

```text
.streamlit/secrets.toml
```

ใส่:

```toml
GROQ_API_KEY = "your-real-groq-api-key"
```

ห้าม commit ไฟล์นี้ขึ้น GitHub

## Run

```bash
streamlit run app.py
```

## RAG Prompt

Prompt หลักอยู่ในตัวแปร `SYSTEM_PROMPT` ใน `app.py`

หลักการสำคัญ:

1. ใช้ Context เท่านั้น
2. ห้ามเดา
3. ถ้า Context ไม่มีคำตอบให้ตอบ `ไม่พบข้อมูลในเอกสาร`
4. ไม่สร้างราคา เบอร์โทรศัพท์ หรือกฎที่ไม่มีในเอกสาร
5. Source จะแสดงโดยตัวแอป

## การ Deploy บน Streamlit Community Cloud

1. Push repository ขึ้น GitHub
2. เปิด Streamlit Community Cloud
3. Create app
4. เลือก repository
5. เลือก branch `main`
6. เลือกไฟล์ `app.py`
7. เปิด Advanced settings
8. ใส่ Secrets:

```toml
GROQ_API_KEY = "your-real-groq-api-key"
```

9. Deploy
10. เปิด URL จาก Incognito หรือเครื่องอื่นเพื่อทดสอบ

## การทดสอบ

ไฟล์ `test_questions.csv` มีคำถามอย่างน้อย 10 ข้อ และมีคำถามที่ไม่มีคำตอบในเอกสารอย่างน้อย 2 ข้อ

ควรทดสอบ:

- คำถามภาษาไทย
- คำถามภาษาอังกฤษ
- คำถามที่มีคำตอบ
- คำถามที่ไม่มีคำตอบ
- การแสดง Source
- การสนทนาต่อเนื่อง
- เปิดเว็บจากเครื่องอื่น

## Security Checklist

- [ ] ไม่มี API Key ใน `app.py`
- [ ] ไม่มี API Key ใน `README.md`
- [ ] `.streamlit/secrets.toml` อยู่ใน `.gitignore`
- [ ] ตรวจ GitHub ก่อนส่งงาน
- [ ] ตั้ง `GROQ_API_KEY` ใน Streamlit Secrets
- [ ] ทดสอบ URL แบบ Incognito

## ตัวอย่างคำถาม

- ต้องเตรียมอุปกรณ์อะไรสำหรับการกางเต็นท์?
- มีกฎเกี่ยวกับการทิ้งขยะอย่างไร?
- ควรเตรียมตัวก่อนเดินป่าอย่างไร?
- มีสิ่งอำนวยความสะดวกอะไรบ้าง?
- หากเกิดเหตุฉุกเฉินควรทำอย่างไร?
- What should I bring for camping?
- What should hikers do to reduce environmental impact?
- อุทยานมีบริการ Wi-Fi หรือไม่?  ← ควรได้ `ไม่พบข้อมูลในเอกสาร`

## หมายเหตุสำหรับการส่งงาน

ก่อนส่งจริง ให้เพิ่มเอกสารต้นฉบับจาก DNP ให้ครบอย่างน้อย 10 ไฟล์ หรือรวมไม่น้อยกว่า 15,000 ตัวอักษร และปรับ `test_questions.csv` ให้คำตอบตรงกับเอกสารชุดสุดท้าย
