import re
import string
from typing import List, Dict, Any, Iterable, Pattern, Union, Optional
import pdfplumber
import nltk


# -------------------------
# Extra cleaning (your rules)
# -------------------------
def clean_text(text: str) -> str:
    """Aggressively clean NCERT/School-book style PDF text with OCR noise."""
    # Normalize spaces/newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove OCR garbled "CChhaapptteerr" like strings
    text = re.sub(r'C+H*A+P+T+E+R+.*?\d+', '', text, flags=re.IGNORECASE)

    # Remove lines with .indd and timestamps (OCR file tags)
    text = re.sub(r'\.?i+n+d+d+\s*\d+.*?(AM|PM)?', '', text, flags=re.IGNORECASE)

    # Remove publisher footer lines
    text = re.sub(r'Curiosity.*?(Grade|Gr\.a\.d\.e)', '', text, flags=re.IGNORECASE)

    # Remove "Chapter ..." repeated headers (even broken ones)
    text = re.sub(r'Chapter\s+The.*?Solutions', '', text, flags=re.IGNORECASE)

    # Define the core tokens once
    fig_core = r'(?:Fig\.?|Figure)\s*:?\s*\d+(?:[.\-\u2013]\d+)*(?:[A-Za-z])?(?:\([A-Za-z]\))?'
    tab_core = r'(?:Tab\.?|Table)\s*:?\s*\d+(?:[.\-\u2013]\d+)*(?:[A-Za-z])?(?:\([A-Za-z]\))?'

    # 1) Remove unbracketed inline refs like "Fig. 9.10a" or "Table 4.1,"
    text = re.sub(rf'\b(?:{fig_core}|{tab_core})(?:\s*[:.,;])?', '', text, flags=re.IGNORECASE)

    # 2) Remove bracketed inline refs like "(Fig. 9.10a)" or "(Figure 3(b))"
    text = re.sub(rf'\(\s*(?:{fig_core}|{tab_core})\s*\)(?:\s*[:.,;])?', '', text, flags=re.IGNORECASE)

    # 3) Remove full caption lines starting with these tokens (with or without brackets)
    text = re.sub(rf'(?mi)^\s*(?:\(\s*)?(?:{fig_core}|{tab_core})(?:\s*\))?\s+.*$', '', text, flags=re.IGNORECASE)

    # 1) Dates like 12/10/2021 or 12-10-21
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)

    # 2) Time like 12:30 or 12:30:45
    text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', text)

    # 3) Weird ::0066::3366 blobs (keep the colons, drop number blobs between them)
    text = re.sub(r'::\d+::', '::', text)

    # 4) Standalone page-like counters with lots of slashes (//2288// style)
    text = re.sub(r'/{2,}\d+/{2,}', ' ', text)

    # 5) Hyphenated index codes like 2020-001-33 at line edges (optional, be careful)
    text = re.sub(r'\b\d{4}-\d{1,3}-\d{1,3}\b', '', text)

    # Collapse multiple punctuation (.... → . , ??? → ? , !!! → !)
    text = re.sub(r'([.?!])\1+', r'\1', text)

    text = re.sub(r"\(\)", "", text)

    # Remove bullets/list markers (normalized to valid escapes; kept your 'z' if it's an OCR bullet)
    text = re.sub(r'[\x8b•·●\-\–\—»"z]', '', text)

    # Normalize ligatures
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')

    return text.strip()


# -------------------------
# Layout helpers
# -------------------------
def words_to_lines(words: List[Dict[str, Any]], line_tol: float = 3.0):
    """
    Group word dicts (from pdfplumber.extract_words) into visual lines by y (top) coordinate.
    Returns a list of {'text','x0','x1','top','bottom'} for each line.
    """
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: List[Dict[str, Any]] = []

    current: List[Dict[str, Any]] = [words_sorted[0]]
    current_top = words_sorted[0]["top"]

    for w in words_sorted[1:]:
        if abs(w["top"] - current_top) <= line_tol:
            current.append(w)
        else:
            lines.append(_make_line(current))
            current = [w]
            current_top = w["top"]

    if current:
        lines.append(_make_line(current))

    return lines


def _make_line(ws: List[Dict[str, Any]]) -> Dict[str, Any]:
    ws_sorted = sorted(ws, key=lambda w: w["x0"])
    text = " ".join(w["text"] for w in ws_sorted)
    return {
        "text": re.sub(r"\s+", " ", text).strip(),
        "x0": min(w["x0"] for w in ws_sorted),
        "x1": max(w["x1"] for w in ws_sorted),
        "top": min(w["top"] for w in ws_sorted),
        "bottom": max(w["bottom"] for w in ws_sorted),
    }


# -------------------------
# Rules and regexes
# -------------------------
# Caption detector (used only if you choose to drop caption lines early)
CAPTION_RE = re.compile(
    r"^\s*(?:\(\s*)?(?:fig(?:ure)?|tab(?:le)?)\.?\s*\d+(?:[.\-–]\d+)*(?:[A-Za-z])?(?:\([A-Za-z]\))?\s*[:\-–)]?\s*",
    re.IGNORECASE,
)

# End-of-sentence punctuation (optionally closed by quotes/paren)
END_PUNCT_RE = re.compile(r'[.!?]["”’)\]]?$')

# Exceptions like "Fig. 10.2", "Sec. 2.3", "No. 5.1" (do not treat as sentence end)
ABBR_BEFORE_NUMBER_RE = re.compile(r"\b(?:fig|sec|no|vol|ch|pg|pp)\.\s*\d", re.IGNORECASE)


def _compile_patterns(patterns: Iterable[Union[str, Pattern]]) -> List[Pattern]:
    out = []
    for p in patterns:
        if isinstance(p, str):
            out.append(re.compile(p, re.IGNORECASE))
        else:
            out.append(p)
    return out


def _chapter_name_to_pattern(name: str) -> Pattern:
    """
    Turn a literal chapter name into a robust regex that matches with flexible punctuation/whitespace.
    """
    tokens = re.findall(r"\w+", name, flags=re.UNICODE)
    if not tokens:
        return re.compile(re.escape(name), re.IGNORECASE)
    pattern = r"\b" + r"[\s\W]*".join(re.escape(tok) for tok in tokens) + r"\b"
    return re.compile(pattern, re.IGNORECASE)


def _build_banned_res(
    banned_line_patterns: Iterable[Union[str, Pattern]],
    chapter_names: Optional[Iterable[str]] = None,
) -> List[Pattern]:
    res = _compile_patterns(banned_line_patterns)
    if chapter_names:
        for name in chapter_names:
            if name and name.strip():
                res.append(_chapter_name_to_pattern(name.strip()))
    return res


# -------------------------
# Main extractor
# -------------------------
def extract_sentences_in_y_range(
    pdf_path: str,
    y_min: float = 100,
    y_max: float = 700,
    line_tol: float = 3.0,
    banned_line_patterns: Iterable[Union[str, Pattern]] = (r"\bgrade\s*8\b",),
    chapter_names: Optional[Iterable[str]] = None,
    remove_strategy: str = "strip",   # "strip" (remove matches within the line) or "drop" (drop whole line)
    drop_caption_lines: bool = True,  # drop lines that look like figure/table captions
    apply_extra_cleaning: bool = True, # apply clean_text() on flushed sentences
    prune_meaningful: bool = False    # NEW: if True, prune sentences that look meaningless/noisy
) -> List[Dict[str, Any]]:
    """
    Extract sentences with layout-aware logic and aggressive cleaning.

    Returns list of:
      {
        "page_number": int,
        "text": str,
        "type": "body",
        "bbox": [x0, top, x1, bottom]
      }
    Note: when drop_caption_lines=True, captions are removed; if False, they are emitted as type="caption".
    """
    results: List[Dict[str, Any]] = []
    banned_res = _build_banned_res(banned_line_patterns, chapter_names)

    def cleanse_line_text(line_text: str) -> str:
        """Remove banned substrings; collapse whitespace."""
        t = line_text
        for pat in banned_res:
            t = pat.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def line_contains_banned(line_text: str) -> bool:
        return any(p.search(line_text) for p in banned_res)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):  # page_idx now starts from 0
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True,
            )

            # Filter by y-range first
            words = [w for w in words if y_min <= float(w["top"]) <= y_max]
            if not words:
                continue

            lines = words_to_lines(words, line_tol=line_tol)
            if not lines:
                continue

            processed_lines: List[Dict[str, Any]] = []
            for ln in lines:
                raw_text = ln["text"]
                if not raw_text:
                    continue

                # Drop caption lines early if requested
                if drop_caption_lines and CAPTION_RE.match(raw_text):
                    continue

                # Banned text removal/drop on a per-line basis
                if remove_strategy == "drop":
                    if line_contains_banned(raw_text):
                        continue
                    new_text = raw_text
                else:
                    new_text = cleanse_line_text(raw_text)
                    if not new_text:
                        continue

                processed_lines.append(
                    {
                        "text": new_text,
                        "x0": ln["x0"],
                        "x1": ln["x1"],
                        "top": ln["top"],
                        "bottom": ln["bottom"],
                        "page_number": page_idx,  # Store page number for each line
                    }
                )

            if not processed_lines:
                continue

            # Build sentences across lines
            buffer_text = ""
            buffer_bbox = None  # bbox of accumulated lines
            buffer_page_number = None  # page number for the current buffer

            def flush_buffer(_type="body"):
                nonlocal buffer_text, buffer_bbox, buffer_page_number
                s = buffer_text.strip()
                if not s:
                    buffer_text = ""
                    buffer_bbox = None
                    buffer_page_number = None
                    return

                # Final cleaning at sentence level
                if apply_extra_cleaning:
                    s = clean_text(s)
                    s = re.sub(r"\s+", " ", s).strip()

                if s:
                    results.append(
                        {
                            "page_number": buffer_page_number if buffer_page_number is not None else 0,
                            "text": s,
                            "type": _type,
                            "bbox": buffer_bbox if buffer_bbox else [0, 0, 0, 0],
                        }
                    )
                buffer_text = ""
                buffer_bbox = None
                buffer_page_number = None

            for ln in processed_lines:
                text = ln["text"]
                if not text:
                    continue

                # Hyphenated line-break fix: if buffer ends with "-", merge without space
                if buffer_text and buffer_text.endswith("-"):
                    buffer_text = buffer_text[:-1] + text.lstrip()
                else:
                    buffer_text = (buffer_text + " " + text).strip() if buffer_text else text

                # Grow bbox
                if buffer_bbox is None:
                    buffer_bbox = [ln["x0"], ln["top"], ln["x1"], ln["bottom"]]
                else:
                    buffer_bbox = [
                        min(buffer_bbox[0], ln["x0"]),
                        min(buffer_bbox[1], ln["top"]),
                        max(buffer_bbox[2], ln["x1"]),
                        max(buffer_bbox[3], ln["bottom"]),
                    ]

                # Track page number for the buffer (use the first line's page number)
                if buffer_page_number is None:
                    buffer_page_number = ln["page_number"]

                # Sentence boundary detection on the (line-level) text
                if END_PUNCT_RE.search(text):
                    tail = text[-12:]
                    if not ABBR_BEFORE_NUMBER_RE.search(tail):
                        flush_buffer("body")

            # Flush trailing text on the page
            flush_buffer("body")

    if prune_meaningful:
        filtered = [r for r in results if _is_meaningful_sentence(r["text"])]
        return filtered

    return results


def _is_meaningful_sentence(s: str, min_words: int = 3, min_alpha_chars: int = 3) -> bool:
    """
    Heuristic checks to prune short/noisy OCR lines that carry no meaningful content.
    Rules:
      - must have at least `min_words` words (simple token split)
      - must contain at least `min_alpha_chars` alphabetic characters
      - must not be punctuation/ellipsis only
      - must not be timestamp-like or repetition like '151 PM PM'
      - must not have a very high punctuation ratio
    """
    if not s:
        return False

    s_stripped = s.strip()
    # drop lines that are just punctuation or repeated dots
    punct_class = re.escape(string.punctuation)
    if re.fullmatch(fr'^[\s{punct_class}]+$', s_stripped):
        return False

    words = [w for w in re.split(r'\s+', s_stripped) if w]
    if len(words) < min_words:
        return False

    alpha_chars = len(re.findall(r'[A-Za-z]', s_stripped))
    if alpha_chars < min_alpha_chars:
        return False

    # punctuation ratio
    punct_chars = len(re.findall(r'[^\w\s]', s_stripped))
    if len(s_stripped) > 0 and (punct_chars / max(1, len(s_stripped)) > 0.6):
        return False

    # timestamp-like or repeated AM/PM noise (e.g., "151 PM PM" or "12:30 PM")
    if re.search(r'\b\d{1,3}(:\d{2})?\s*(?:AM|PM)\b', s_stripped, flags=re.IGNORECASE):
        # if there's more alphabetic text around the timestamp it's ok; otherwise drop
        # consider it noise if AM/PM occurs >=2 times or if only digits+AM/PM present
        ampm_count = len(re.findall(r'\b(?:AM|PM)\b', s_stripped, flags=re.IGNORECASE))
        if ampm_count >= 2 or re.fullmatch(r'^[\d\s:APMapm.]+$', s_stripped):
            return False

    # common short OCR fragments and single-character noise
    if re.fullmatch(r'[\.\-]+', s_stripped):
        return False

    # drop lines with excessive digit-only content
    digit_chars = len(re.findall(r'\d', s_stripped))
    if digit_chars > 0 and (digit_chars / max(1, len(s_stripped)) > 0.6) and alpha_chars < 2:
        return False

    return True

def extract_metadata_from_pdf(path: str, names_to_ignore: list[str]) -> List[str]:
    # Adjust y-range to exclude headers/footers if needed
    y_min, y_max = 20, 1000
    sentences = extract_sentences_in_y_range(
        path,
        y_min=y_min,
        y_max=y_max,
        line_tol=3.0,
        banned_line_patterns=names_to_ignore,
        chapter_names=names_to_ignore[:1],
        remove_strategy="strip",
        drop_caption_lines=True,
        apply_extra_cleaning=True,
        prune_meaningful=True,   # enable pruning of non-meaningful lines
    )

    return chunk_text_list([s["text"] for s in sentences])

def chunk_text_list(text_list: List[str], n=100) -> List[str]:
    """
    Takes a list of strings (e.g., book pages/paragraphs),
    cleans them, concatenates, and chunks into ~n-word blocks.
    Returns a list of chunks.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')

    # Join all input strings into one big text
    full_text = " ".join(text_list)

    sentences = nltk.sent_tokenize(full_text)

    chunks, cur_chunk = [], ""

    for sentence in sentences:
        cur_chunk += " " + sentence
        if len(cur_chunk.split()) >= n:
            chunks.append(cur_chunk.strip())
            cur_chunk = ""

    if cur_chunk:
        chunks.append(cur_chunk.strip())

    return chunks


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    pdf_path = r"./pdfs/Class 8 Science CH 9.pdf"  # Replace with your PDF file path

    chapter_names = [
        "9.1 What Are Solute, Solvent, and Solution?",
    ]

    # Extend banned patterns as needed (hidden grade/class marks, roman etc.)
    banned = [
        r"\bgrade\s*8\b",
        r"\bclass\s*8\b",
        r"\bclass\s*viii\b",
    ]

    sentences = extract_metadata_from_pdf(pdf_path, names_to_ignore=banned)

    for s in sentences:
        print(s)
        print()

