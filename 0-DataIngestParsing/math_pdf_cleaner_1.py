# ============================================================
# 📐 Math PDF Cleaner for RAG Pipeline
# LangChain (PyMuPDFLoader) for text + pdfplumber for tables
# No fitz/PyMuPDF used directly — all via LangChain
# ============================================================

import re
import os
import json
import pdfplumber
from langchain_community.document_loaders import PyMuPDFLoader

PDF_PATH = "data/pdf/Trignometry.pdf"
OUTPUT_DIR = "data/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 🔧 CORE CLEANING FUNCTION
# ============================================================

def clean_math_text(text: str) -> str:
    """
    Cleans raw PDF-extracted text with broken math formulas.
    Handles all artifacts from LangChain PyMuPDFLoader extraction.
    """
    if not text:
        return ""

    # ── 1. Greek letters on isolated lines ─────────────────────────────
    # e.g. "\nπ\n" → " π "
    text = re.sub(r'\n([πΠθΘαβγδεζηλμσφψωΩ∑∫√±×÷])\n', r' \1 ', text)

    # ── 2. Broken fractions with dash line ─────────────────────────────
    # e.g. "π\n─\n180" → "π/180"
    text = re.sub(r'([^\n]+)\n[─—━]\n([^\n]+)', r'\1/\2', text)

    # ── 3. π/number: "π\n180" → "π/180" ───────────────────────────────
    text = re.sub(r'\bπ\n(\d+)\b', r'π/\1', text)

    # ── 4. number/π: "180 π °" → "(180/π)°" ───────────────────────────
    text = re.sub(r'(\d+)\s+π\s*°', r'(\1/π)°', text)

    # ── 5. Split degree symbols: "45\n°" → "45°" ───────────────────────
    text = re.sub(r'(\d+)\s*\n\s*°', r'\1°', text)

    # ── 6. Orphaned operators on their own lines ────────────────────────
    # e.g. "\n=\n" → " = "
    text = re.sub(r'\n([=×÷+\-<>≤≥≠])\n', r' \1 ', text)

    # ── 7. Formula line breaks ──────────────────────────────────────────
    # e.g. "θ =\n1" → "θ = 1"
    text = re.sub(r'([=+\-×÷])\n([0-9πθ√(])', r'\1 \2', text)

    # ── 8. Mixed numbers: "1 1\n2" → "1 1/2" ───────────────────────────
    text = re.sub(r'(\d+)\s+(\d+)\n(\d+)', r'\1 \2/\3', text)

    # ── 9. Numeric fractions: "22\n7" → "22/7" ─────────────────────────
    text = re.sub(r'\b(\d+)\n(\d+)\b', r'\1/\2', text)

    # ── 10. Letter fractions: "l\nr" → "l/r" ───────────────────────────
    text = re.sub(r'\b([a-zA-Zθφλ])\n([a-zA-Zθφλ0-9])\b', r'\1/\2', text)

    # ── 11. sin/cos split: "sin\n²θ" → "sin²θ" ─────────────────────────
    text = re.sub(r'(sin|cos|tan|cot|sec|cosec)\s*\n\s*([²³]?\s*[θαβγ])', r'\1\2', text)

    # ── 12. Radian/Degree measure formulas ─────────────────────────────
    text = re.sub(
        r'Radian\s+measure\s*=\s*π\s*[\n\s]*180\s*[\n\s]*[×x]\s*Degree\s+measure',
        'Radian measure = (π/180) × Degree measure',
        text
    )
    text = re.sub(
        r'Degree\s+measure\s*=\s*180\s*[\n\s]*π\s*[\n\s]*[×x]\s*Radian\s+measure',
        'Degree measure = (180/π) × Radian measure',
        text
    )

    # ── 13. Trig table fix (linearized by PyMuPDF) ─────────────────────
    text = re.sub(
        r'Radian\s+π\s+6\s+π\s+4\s+π\s+3\s+π\s+2\s+π\s+3π\s*\n?\s*2\s+2π',
        'Radian: π/6  π/4  π/3  π/2  π  3π/2  2π',
        text
    )

    # ── 14. Excess blank lines and spaces ───────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # ── 15. Strip trailing whitespace per line ──────────────────────────
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# ============================================================
# 📄 STEP 1 — LangChain PyMuPDFLoader: Extract Text
# ============================================================

print("\n1️⃣  LangChain PyMuPDFLoader — Text Extraction")
print("─" * 55)

langchain_docs = []

try:
    loader = PyMuPDFLoader(PDF_PATH)
    raw_docs = loader.load()
    print(f"  Loaded {len(raw_docs)} pages")

    for doc in raw_docs:
        # Clean the page_content in-place, metadata stays untouched
        doc.page_content = clean_math_text(doc.page_content)
        langchain_docs.append(doc)

    print(f"  ✅ Cleaned {len(langchain_docs)} pages")
    print(f"\n  Preview — Page 1:")
    print(f"  {langchain_docs[0].page_content[:300]}...")
    print(f"\n  Metadata: {langchain_docs[0].metadata}")

except Exception as e:
    print(f"  ❌ Error: {e}")


# ============================================================
# 📊 STEP 2 — pdfplumber: Extract Tables
# ============================================================

print("\n2️⃣  pdfplumber — Table Extraction")
print("─" * 55)

all_tables = {}   # { page_number (1-indexed): [ [cleaned_row], ... ] }

try:
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"  Scanning {len(pdf.pages)} pages for tables...")

        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()   # all tables on this page

            if tables:
                print(f"\n  Page {page_num + 1}: Found {len(tables)} table(s)")
                all_tables[page_num + 1] = []

                for t_idx, table in enumerate(tables):
                    print(f"\n    Table {t_idx + 1}:")
                    cleaned_table = []

                    for row in table:
                        cleaned_row = [
                            clean_math_text(cell.strip()) if cell else ""
                            for cell in row
                        ]
                        cleaned_table.append(cleaned_row)
                        print(f"      {cleaned_row}")

                    all_tables[page_num + 1].append(cleaned_table)
            else:
                print(f"  Page {page_num + 1}: no tables")

    print(f"\n  ✅ Tables found on pages: {list(all_tables.keys())}")

except Exception as e:
    print(f"  ❌ Error: {e}")


# ============================================================
# 🔀 STEP 3 — Merge: Inject Tables into LangChain Docs
# Replaces linearized table garbage in page_content
# with properly structured table from pdfplumber
# ============================================================

print("\n3️⃣  Merging Tables into LangChain Docs")
print("─" * 55)

def table_to_text(table: list) -> str:
    """Converts a pdfplumber table (list of rows) to readable pipe-delimited text."""
    lines = []
    for row in table:
        lines.append(" | ".join(cell for cell in row))
    return "\n".join(lines)


for doc in langchain_docs:
    # LangChain metadata "page" is 0-indexed; pdfplumber is 1-indexed
    page_num = doc.metadata.get("page", None)
    page_key = page_num + 1 if page_num is not None else None

    if page_key and page_key in all_tables:
        table_text = ""
        for i, table in enumerate(all_tables[page_key]):
            table_text += f"\n[TABLE {i+1}]\n"
            table_text += table_to_text(table)
            table_text += "\n"

        # Append structured table to page content
        doc.page_content += f"\n\n{table_text}"
        print(f"  Page {page_key}: injected {len(all_tables[page_key])} table(s)")

print(f"\n  ✅ Merge complete")


# ============================================================
# 🧩 STEP 4 — RAG-Ready Chunking
# Splits at natural math boundaries — never mid-formula
# ============================================================

print("\n4️⃣  RAG Chunking")
print("─" * 55)

def chunk_for_rag(docs: list, chunk_size: int = 500) -> list:
    """
    Splits LangChain docs into RAG-ready chunks.
    Respects headings, Example N, Theorem, Definition boundaries.
    """
    chunks = []
    chunk_id = 0

    split_pattern = re.compile(
        r'(?=\n(?:Example\s+\d+|Theorem|Definition|Note|Solution|'
        r'Exercise|Chapter\s+\d+|EXERCISE|SUMMARY|[A-Z][A-Z\s]{4,})\b)',
        re.IGNORECASE
    )

    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        full_text = f"[Page {page_num}]\n" + doc.page_content

        sections = split_pattern.split(full_text)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "content": section,
                    "length": len(section)
                })
                chunk_id += 1
            else:
                # Split long sections by paragraph
                paragraphs = re.split(r'\n\n+', section)
                buffer = ""
                for para in paragraphs:
                    if len(buffer) + len(para) < chunk_size:
                        buffer += "\n\n" + para
                    else:
                        if buffer.strip():
                            chunks.append({
                                "chunk_id": chunk_id,
                                "page": page_num,
                                "content": buffer.strip(),
                                "length": len(buffer.strip())
                            })
                            chunk_id += 1
                        buffer = para
                if buffer.strip():
                    chunks.append({
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "content": buffer.strip(),
                        "length": len(buffer.strip())
                    })
                    chunk_id += 1

    return chunks


rag_chunks = chunk_for_rag(langchain_docs)
print(f"  Total chunks: {len(rag_chunks)}")

for c in rag_chunks[:3]:
    print(f"\n  --- Chunk {c['chunk_id']} | Page {c['page']} ({c['length']} chars) ---")
    print(f"  {c['content'][:200]}...")


# ============================================================
# 💾 STEP 5 — Save Outputs
# ============================================================

print("\n5️⃣  Saving Outputs")
print("─" * 55)

# Cleaned text (all pages)
text_out = os.path.join(OUTPUT_DIR, "langchain_cleaned.txt")
with open(text_out, "w", encoding="utf-8") as f:
    for doc in langchain_docs:
        page_num = doc.metadata.get("page", 0) + 1
        f.write(f"\n{'='*60}\n")
        f.write(f"PAGE {page_num} | {doc.metadata}\n")
        f.write(f"{'='*60}\n")
        f.write(doc.page_content)
        f.write("\n")
print(f"  ✅ Cleaned text : {text_out}")

# Tables as JSON
tables_out = os.path.join(OUTPUT_DIR, "tables.json")
with open(tables_out, "w", encoding="utf-8") as f:
    json.dump(all_tables, f, ensure_ascii=False, indent=2)
print(f"  ✅ Tables JSON  : {tables_out}")

# RAG chunks as JSON — feed directly into your vector DB
chunks_out = os.path.join(OUTPUT_DIR, "rag_chunks.json")
with open(chunks_out, "w", encoding="utf-8") as f:
    json.dump(rag_chunks, f, ensure_ascii=False, indent=2)
print(f"  ✅ RAG chunks   : {chunks_out}")


# ============================================================
# 📊 SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅  PIPELINE COMPLETE")
print("=" * 60)
print(f"  PDF        : {PDF_PATH}")
print(f"  Pages      : {len(langchain_docs)}")
print(f"  Tables     : {sum(len(v) for v in all_tables.values())} table(s) on {len(all_tables)} page(s)")
print(f"  RAG Chunks : {len(rag_chunks)}")
print(f"\n  Output Files:")
print(f"    📄 {text_out}")
print(f"    📊 {tables_out}")
print(f"    🧩 {chunks_out}  ← Feed into your vector DB")
print("=" * 60)
