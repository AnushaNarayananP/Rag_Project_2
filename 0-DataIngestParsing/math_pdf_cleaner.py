# ============================================================
# 📐 Math PDF Cleaner for RAG Pipeline
# Handles trigonometry & math textbook PDFs with broken formulas
# ============================================================

import re
import os
import json
import fitz  # PyMuPDF
import pdfplumber

PDF_PATH = "data/pdf/Trignometry.pdf"
OUTPUT_DIR = "data/cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 🔧 CORE CLEANING FUNCTION
# ============================================================

def clean_math_text(text: str) -> str:
    """
    Cleans raw PDF-extracted text that contains broken math formulas.
    Handles LaTeX-style artifacts from pdfplumber / PyMuPDF extraction.
    """
    if not text:
        return ""

    # ── 1. Fix broken Greek letters on isolated lines ──────────────────
    # e.g. "\nπ\n180"  →  " π 180"  then fraction fix handles it
    text = re.sub(r'\n([πΠθΘαβγδεζηλμσφψωΩ∑∫√±×÷])\n', r' \1 ', text)

    # ── 2. Collapse broken fraction patterns ───────────────────────────
    # e.g. "π\n─\n180"  →  "π/180"
    text = re.sub(r'([^\n]+)\n[─—━]\n([^\n]+)', r'\1/\2', text)

    # ── 3. Fix degree symbols split from numbers ────────────────────────
    # e.g. "45\n°"  →  "45°"
    text = re.sub(r'(\d+)\s*\n\s*°', r'\1°', text)

    # ── 4. Fix radian / degree measure split lines ──────────────────────
    # e.g. "Radian measure = π\n180 × Degree"  →  keeps on same line
    text = re.sub(r'(measure\s*=\s*[^\n]*)\n(\s*[×÷]\s*)', r'\1 \2', text)

    # ── 5. Remove orphaned math operators on their own lines ────────────
    # e.g. "\n=\n"  →  " = "
    text = re.sub(r'\n([=×÷+\-<>≤≥≠∈∉⊂⊃])\n', r' \1 ', text)

    # ── 6. Rejoin lines that are clearly part of the same formula ───────
    # A line ending with = and next line starting with a number/symbol
    text = re.sub(r'([=+\-×÷])\n([0-9πθ√(])', r'\1 \2', text)

    # ── 7. Normalize "Radian measure = π / 180 × Degree measure" ────────
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

    # ── 8. Fix sin²θ, cos²θ patterns that get split ─────────────────────
    text = re.sub(r'(sin|cos|tan|cot|sec|cosec)\s*\n\s*([²³]?\s*[θαβγ])', r'\1\2', text)

    # ── 9. Collapse 3+ blank lines into 2 ───────────────────────────────
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ── 10. Normalize multiple spaces ───────────────────────────────────
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # ── 11. Strip trailing whitespace per line ───────────────────────────
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# ============================================================
# 📄 METHOD 1 — pdfplumber (Good for text-heavy PDFs)
# ============================================================

print("\n1️⃣  pdfplumber Extraction")
print("─" * 50)

pdfplumber_pages = []

try:
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"  Loaded {len(pdf.pages)} pages")

        for i, page in enumerate(pdf.pages):
            raw = page.extract_text()

            if raw:
                cleaned = clean_math_text(raw)
                pdfplumber_pages.append({
                    "page": i + 1,
                    "raw_length": len(raw),
                    "cleaned_length": len(cleaned),
                    "content": cleaned
                })
                print(f"  Page {i+1}: {len(raw)} chars → {len(cleaned)} chars (cleaned)")
            else:
                print(f"  Page {i+1}: ⚠️  No text extracted (may be image-based)")

    # Save pdfplumber output
    plumber_out = os.path.join(OUTPUT_DIR, "pdfplumber_cleaned.txt")
    with open(plumber_out, "w", encoding="utf-8") as f:
        for p in pdfplumber_pages:
            f.write(f"\n{'='*60}\n")
            f.write(f"PAGE {p['page']}\n")
            f.write(f"{'='*60}\n")
            f.write(p["content"])
            f.write("\n")

    print(f"\n  ✅ Saved: {plumber_out}")

except Exception as e:
    print(f"  ❌ Error: {e}")


# ============================================================
# 📄 METHOD 2 — PyMuPDF / fitz (Fast, accurate, rich metadata)
# ============================================================

print("\n2️⃣  PyMuPDF (fitz) Extraction")
print("─" * 50)

pymupdf_pages = []

try:
    doc = fitz.open(PDF_PATH)
    print(f"  Loaded {len(doc)} pages")
    print(f"  Metadata: {doc.metadata}")

    for i, page in enumerate(doc):
        raw = page.get_text("text")  # plain text mode

        if raw:
            cleaned = clean_math_text(raw)
            pymupdf_pages.append({
                "page": i + 1,
                "metadata": {
                    "page_number": i + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation
                },
                "raw_length": len(raw),
                "cleaned_length": len(cleaned),
                "content": cleaned
            })
            print(f"  Page {i+1}: {len(raw)} chars → {len(cleaned)} chars (cleaned)")
        else:
            print(f"  Page {i+1}: ⚠️  No text found")

    doc.close()

    # Save PyMuPDF output
    pymupdf_out = os.path.join(OUTPUT_DIR, "pymupdf_cleaned.txt")
    with open(pymupdf_out, "w", encoding="utf-8") as f:
        for p in pymupdf_pages:
            f.write(f"\n{'='*60}\n")
            f.write(f"PAGE {p['page']} | {p['metadata']}\n")
            f.write(f"{'='*60}\n")
            f.write(p["content"])
            f.write("\n")

    print(f"\n  ✅ Saved: {pymupdf_out}")

except Exception as e:
    print(f"  ❌ Error: {e}")


# ============================================================
# 🔀 METHOD 3 — Smart Merge: Best-of-both extraction
# Picks whichever method recovered more text per page
# ============================================================

print("\n3️⃣  Smart Merge (Best-of-both extraction)")
print("─" * 50)

merged_pages = []
total_pages = max(len(pdfplumber_pages), len(pymupdf_pages))

for i in range(total_pages):
    plumber_text = pdfplumber_pages[i]["content"] if i < len(pdfplumber_pages) else ""
    pymupdf_text = pymupdf_pages[i]["content"] if i < len(pymupdf_pages) else ""

    # Pick the extraction that recovered more text
    if len(pymupdf_text) >= len(plumber_text):
        chosen = pymupdf_text
        source = "pymupdf"
    else:
        chosen = plumber_text
        source = "pdfplumber"

    merged_pages.append({
        "page": i + 1,
        "source": source,
        "content": chosen
    })
    print(f"  Page {i+1}: used {source} ({len(chosen)} chars)")

# Save merged output
merged_out = os.path.join(OUTPUT_DIR, "merged_cleaned.txt")
with open(merged_out, "w", encoding="utf-8") as f:
    for p in merged_pages:
        f.write(f"\n{'='*60}\n")
        f.write(f"PAGE {p['page']} [source: {p['source']}]\n")
        f.write(f"{'='*60}\n")
        f.write(p["content"])
        f.write("\n")

print(f"\n  ✅ Saved: {merged_out}")


# ============================================================
# 🧩 METHOD 4 — RAG-Ready Chunking
# Splits by section headings, examples, theorems
# Never cuts mid-formula or mid-example
# ============================================================

print("\n4️⃣  RAG Chunking")
print("─" * 50)

def chunk_for_rag(pages: list, chunk_size: int = 500) -> list:
    """
    Splits cleaned pages into RAG-friendly chunks.
    Respects natural math boundaries: headings, Example N, Theorem, etc.
    """
    chunks = []
    chunk_id = 0

    # Patterns that indicate a clean split point
    split_pattern = re.compile(
        r'(?=\n(?:Example\s+\d+|Theorem|Definition|Note|Solution|Exercise|'
        r'Chapter\s+\d+|EXERCISE|SUMMARY|[A-Z][A-Z\s]{4,})\b)',
        re.IGNORECASE
    )

    # Build full text with page markers
    full_text = ""
    for p in pages:
        full_text += f"\n\n[Page {p['page']}]\n" + p["content"]

    # Split on natural section boundaries
    sections = split_pattern.split(full_text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If section is small enough, keep as-is
        if len(section) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "content": section,
                "length": len(section)
            })
            chunk_id += 1
        else:
            # Further split long sections by paragraph
            paragraphs = re.split(r'\n\n+', section)
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) < chunk_size:
                    buffer += "\n\n" + para
                else:
                    if buffer.strip():
                        chunks.append({
                            "chunk_id": chunk_id,
                            "content": buffer.strip(),
                            "length": len(buffer.strip())
                        })
                        chunk_id += 1
                    buffer = para
            if buffer.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": buffer.strip(),
                    "length": len(buffer.strip())
                })
                chunk_id += 1

    return chunks


rag_chunks = chunk_for_rag(merged_pages)
print(f"  Total chunks created: {len(rag_chunks)}")

# Preview first 3 chunks
for c in rag_chunks[:3]:
    print(f"\n  --- Chunk {c['chunk_id']} ({c['length']} chars) ---")
    print(f"  {c['content'][:200]}...")

# Save chunks as JSON — ready to pass to your embedding model
chunks_out = os.path.join(OUTPUT_DIR, "rag_chunks.json")
with open(chunks_out, "w", encoding="utf-8") as f:
    json.dump(rag_chunks, f, ensure_ascii=False, indent=2)

print(f"\n  ✅ Saved {len(rag_chunks)} chunks: {chunks_out}")


# ============================================================
# 📊 FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅  PIPELINE COMPLETE")
print("=" * 60)
print(f"  PDF            : {PDF_PATH}")
print(f"  Pages          : {total_pages}")
print(f"  RAG Chunks     : {len(rag_chunks)}")
print(f"\n  Output Files:")
print(f"    📄 {plumber_out}")
print(f"    📄 {pymupdf_out}")
print(f"    📄 {merged_out}")
print(f"    🧩 {chunks_out}  ← Feed this into your vector DB")
print("=" * 60)
