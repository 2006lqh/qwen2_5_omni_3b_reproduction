from pathlib import Path
import os
import re


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PDF = Path(os.environ.get("MASQUANT_PAPER_PDF", "MASQuant.pdf"))
OUT_DIR = ROOT / "results/paper_benchmark_review_20260524"


def extract_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def compact(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def snippets(text: str, terms: list[str], window: int = 600) -> list[tuple[str, str]]:
    hits = []
    lowered = text.lower()
    for term in terms:
        start = 0
        term_lower = term.lower()
        while True:
            idx = lowered.find(term_lower, start)
            if idx < 0:
                break
            hits.append((term, text[max(0, idx - window) : idx + len(term) + window].strip()))
            start = idx + len(term)
            if sum(1 for hit_term, _ in hits if hit_term == term) >= 8:
                break
    return hits


def main():
    if not PDF.exists():
        raise SystemExit(f"missing PDF: {PDF}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = compact(extract_text(PDF))
    (OUT_DIR / "paper_full_text_extracted.txt").write_text(text, encoding="utf-8")

    terms = [
        "benchmark",
        "dataset",
        "evaluation",
        "LibriSpeech",
        "WenetSpeech",
        "MMMU",
        "OmniBench",
        "OCRBench",
        "TextVQA",
        "VizWiz",
        "ScienceQA",
        "Table 1",
        "Table 2",
        "Table 3",
        "WER",
        "accuracy",
    ]
    hits = snippets(text, terms)
    seen = set()
    blocks = []
    for term, snippet in hits:
        key = (term, snippet[:200])
        if key in seen:
            continue
        seen.add(key)
        blocks.append(f"## {term}\n\n{snippet}\n")
    (OUT_DIR / "paper_benchmark_snippets.md").write_text("\n".join(blocks), encoding="utf-8")
    print(f"text_chars={len(text)}")
    print(f"found_terms={','.join(sorted({term for term, _ in hits}))}")
    print(f"out_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
