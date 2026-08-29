# Data Extraction Pipelines & Document RAG Agent

Document-extraction and retrieval work for pharmaceutical PDFs: getting reliable text and
structured fields out of both digital and scanned documents, then making that text
answerable through a retrieval pipeline.

Pharmaceutical suppliers issue large volumes of controlled documents — certificates of
quality, certificates of analysis, regulatory statements. Some are digital PDFs with a real
text layer. Many are scans, where the text is only pixels. This repo holds the extraction
work for both cases, plus the OCR engine evaluation used to decide which tool handles which
document class.

Built during an externship in pharmaceutical document processing. Work in progress.

---

## Repository layout

| File | Input type | What it does |
|---|---|---|
| `tesseract_paddle_easy_ocr_base_comparison.py` | Scanned PDF | Benchmarks three OCR engines on the same page |
| `dataprocessing_pipe_sub_2.py` | Scanned PDF | Full OCR pipeline — preprocess, extract, locate, structure |
| `mini_pdfdataextraction_pymupdf.py` | Digital PDF | Field extraction with coordinates, no OCR needed |
| `sample_chatbox.py` | — | Minimal LLM chat loop; the generation half of the RAG work |

---

## `tesseract_paddle_easy_ocr_base_comparison.py`

An engine evaluation, not a demo. The goal is an evidence-based answer to "which OCR engine
for this document class," so all three engines run against an identical input.

**How it works**

1. Prints the runtime environment first — Python version, platform, GPU via `nvidia-smi`,
   CUDA via `nvcc` — because OCR results and speed differ between CPU and GPU runtimes, and
   a benchmark without that recorded isn't reproducible.
2. Renders the PDF to page images at 300 DPI with PyMuPDF, then converts the pixmap to a
   NumPy array by slicing off the alpha channel.
3. Pulls the PDF's own embedded text layer as a **baseline** to compare OCR output against —
   ground truth where the document has one.
4. Runs the same page array through each engine:
   - **Tesseract** — `image_to_data` with `Output.DICT`, filtering out empty tokens and
     `conf == -1` placeholder rows
   - **PaddleOCR** — CPU device, orientation/unwarping/textline-orientation classifiers
     disabled to isolate raw recognition quality; returns `rec_texts`, `rec_scores`, `rec_polys`
   - **EasyOCR** — `readtext`, returns polygon/text/confidence triples
5. **Normalizes all three into one schema** so the comparison is apples-to-apples. Paddle and
   EasyOCR return four-corner polygons; `paddleocr_to_words()` and `easyocr_to_words()`
   convert those to the same `{text, conf, left, top, width, height}` rectangle format
   Tesseract emits natively.
6. `reconstruct_lines()` rebuilds reading order from the flat word list — sorts by vertical
   position, groups words into rows using a tolerance derived from the *median* word height
   (`max(10, median_height * 0.6)`), then sorts each row left-to-right. Median rather than
   mean so one oversized heading doesn't distort the row grouping.
7. Writes `paddle_output.json` and `easyocr_output.json`. `to_native()` unwraps NumPy scalar
   types, which aren't JSON-serializable.

---

## `dataprocessing_pipe_sub_2.py`

The end-to-end pipeline for scanned documents — from PDF bytes to structured JSON with
coordinates. Tesseract only, since this is production path rather than evaluation.

**Preprocessing**

- `get_pages()` — generator, renders pages at 300 DPI one at a time rather than holding every
  page in memory
- `make_gray()` — RGB to grayscale
- `image_thresholding()` — Otsu binarization, which picks the threshold from the image's own
  histogram instead of a fixed cutoff, so it adapts to varying scan brightness
- `noise_filtering()` — currently a pass-through. A bilateral filter is left in place,
  commented, with a note that results were measurably **better without it** — the filter was
  softening character edges Tesseract needed. Kept as a documented negative result.
- `split_sidebar()` — cuts the leftmost 9% of the page. These documents carry a vertical
  margin strip that OCRs into garbage and pollutes downstream regex matches.

**Extraction and correction**

- Tesseract runs with `--oem 3 --psm 6` (default engine, "assume a uniform block of text" —
  correct for dense single-column documents).
- `standard_corrections()` fixes the specific digit/letter confusions this document class
  produces: `L0T` → `LOT`, `C0A` → `COA`, `CERT1FICATE`, `EXP1RATION`, plus stripping stray
  `|`, `_`, `+` artifacts from table rules and scan noise. Targeted substitutions, not a
  spellchecker — a general one would corrupt legitimate part numbers.
- Regex pulls named fields (process run IDs, 8-digit lot numbers) out of the corrected text.

**Localization and output**

- `image_to_data` gives per-word bounding boxes. Coordinates are taken from `main_body` —
  the same image OCR actually ran on — so boxes align with the text instead of being offset
  by the sidebar crop.
- Two visual overlays: every word above confidence 40, and only the words matching
  `key_fields` (CERTIFICATE, LOT, BATCH, VENDOR, EXPIRATION, DATE, SIGNATURE, COA,
  MANUFACTURER).
- Field matching is **substring**, not equality, so `LOT:` and `LOT#` still match `LOT`.
- Final JSON keys each field to a **list** of hits, so a document with several dates doesn't
  silently lose all but the last one. Each entry carries text, bbox, and confidence.

---

## `mini_pdfdataextraction_pymupdf.py`

For digital PDFs that already have a text layer. No OCR — but the hard part remains: a regex
matches against a *string*, while the answer needed is a *location on the page*.

**The core idea**

- `build_positioned_text()` joins PyMuPDF's word tokens into a single searchable string while
  recording, for every word, its character offset range and its bounding box.
- `bbox_for_match()` takes a regex match's character span, finds every word whose offsets
  overlap it, and returns the union of their boxes.

That pairing is what makes the file useful: you write ordinary regex against readable text and
still get page coordinates back, without hand-walking the word list.

**Field handling**

- `FIELD_PATTERNS` — one named pattern per field (vendor, document type, manufacturing date,
  expiration date, revision number, lot number), so adding a field means adding a dict entry.
- `is_real_date()` validates candidates against five real formats via `strptime` before
  accepting them. An 8-digit regex match is not necessarily a date — this rejects part numbers
  that happen to be eight digits.
- `to_iso()` normalizes accepted dates to `YYYY-MM-DD`.
- `extract_table()` — PyMuPDF `find_tables()`, returns bbox plus extracted rows.
- `draw_field_boxes()` — renders the page at 2× zoom and draws labelled boxes. Coordinates are
  scaled by the same zoom factor so they land correctly on the upscaled image.

---

## `sample_chatbox.py`

A minimal multi-turn chat loop over Gemini via LlamaIndex — the generation half of the
retrieval pipeline, isolated so the LLM layer can be verified before retrieval is wired in.

Appends both user and assistant turns to a `ChatMessage` list so context carries across the
conversation, exits on `exit`/`quit`/`bye`, and wraps the call in try/except so one API error
doesn't kill the session.

> The API key is a placeholder. Set your own via environment variable:
> ```bash
> export GOOGLE_API_KEY="your-key-here"
> ```
> Do not paste a real key into the file.

---

## Running these

Each file was written in Google Colab and carries its own `!pip install` / `!apt-get` lines at
the top. To run locally instead:

```bash
pip install pymupdf pytesseract easyocr paddleocr paddlepaddle \
            opencv-python-headless pillow numpy pandas \
            llama-index llama-index-llms-google-genai
```

Tesseract is a system binary, not a Python package:

```bash
# Debian / Ubuntu
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
# macOS
brew install tesseract poppler
```

The scripts expect an input PDF in the working directory — `data.pdf` for the OCR pipeline,
`sample-sdf-document.pdf` for the PyMuPDF extractor. No sample documents are committed to this
repo; the source documents are supplier-controlled and are not redistributed here.

---

## Notes

- **Field patterns are document-specific.** The regexes are tuned to one supplier's document
  layout. Point them at a different vendor's format and they need retuning — that's inherent
  to regex-based extraction, and part of what motivates the RAG half of this project.
- **Notebooks are exported as `.py`.** Colab notebooks embed page images and widget state in
  their outputs, which bloats diffs and can carry document contents into version control.
  Committing the script export keeps the repo clean and keeps source documents out of it.

## Stack

`Python` · `PyMuPDF` · `Tesseract` · `PaddleOCR` · `EasyOCR` · `OpenCV` · `LlamaIndex` ·
`Gemini` · `NumPy` · `regex` · `PDF parsing` · `OCR` · `RAG`
