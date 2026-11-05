# Data Extraction Pipelines & Document RAG Agent

Extraction and retrieval work for pharmaceutical PDFs: getting reliable text and structured fields out of digital and scanned documents, then making that text answerable through a retrieval pipeline.

Pharma suppliers issue large volumes of controlled documents such as certificates of quality, certificates of analysis, and regulatory statements. Some are digital PDFs with a real text layer; many are scans where the text exists only as pixels. This repo covers both cases, plus the OCR engine comparison used to decide which tool handles which kind of document.

Built during an externship in pharmaceutical document processing. Work in progress.

## Files

| File | Input | What it does |
|---|---|---|
| `tesseract_paddle_easy_ocr_base_comparison.py` | Scanned PDF | Benchmarks three OCR engines on the same page |
| `dataprocessing_pipe_sub_2.py` | Scanned PDF | Full OCR pipeline: preprocess, extract, locate, structure |
| `mini_pdfdataextraction_pymupdf.py` | Digital PDF | Field extraction with coordinates, no OCR needed |
| `sample_chatbox.py` | n/a | Minimal LLM chat loop, the generation half of the RAG work |

## OCR engine comparison (`tesseract_paddle_easy_ocr_base_comparison.py`)

The goal is an evidence-based answer to "which OCR engine for this kind of document," so all three engines run on the same input.

1. Prints the runtime first (Python version, platform, GPU via `nvidia-smi`, CUDA via `nvcc`). OCR results and speed differ between CPU and GPU, so a benchmark that doesn't record this can't be reproduced.
2. Renders the PDF to 300 DPI page images with PyMuPDF and converts each pixmap to a NumPy array, dropping the alpha channel.
3. Pulls the PDF's embedded text layer as a baseline to compare OCR output against, where the document has one.
4. Runs the same page through each engine:
   - Tesseract: `image_to_data` with `Output.DICT`, filtering out empty tokens and `conf == -1` placeholder rows
   - PaddleOCR: on CPU, with the orientation, unwarping and text-line orientation classifiers turned off to isolate raw recognition quality
   - EasyOCR: `readtext`, which returns polygon, text and confidence
5. Converts all three outputs to one schema. PaddleOCR and EasyOCR return four-corner polygons; `paddleocr_to_words()` and `easyocr_to_words()` turn them into the `{text, conf, left, top, width, height}` boxes Tesseract produces.
6. `reconstruct_lines()` rebuilds reading order: it sorts words top to bottom, groups them into rows with a tolerance of `max(10, median_height * 0.6)`, then sorts each row left to right. It uses the median word height so one large heading doesn't throw off the grouping.
7. Writes `paddle_output.json` and `easyocr_output.json`, using `to_native()` to unwrap NumPy scalars that JSON can't serialize.

## Scanned-document pipeline (`dataprocessing_pipe_sub_2.py`)

Takes a scanned PDF to structured JSON with coordinates. It uses Tesseract only, since this is the production path rather than an evaluation.

Preprocessing:

- `get_pages()` is a generator that renders one 300 DPI page at a time instead of holding every page in memory.
- `make_gray()` converts RGB to grayscale.
- `image_thresholding()` applies Otsu binarization, which picks the threshold from each image's histogram and so adapts to scans of different brightness.
- `noise_filtering()` is currently a pass-through. A bilateral filter is left in, commented out, because results were measurably better without it: it softened the character edges Tesseract relies on.
- `split_sidebar()` cuts the leftmost 9% of the page. These documents have a vertical margin strip that OCRs into garbage and breaks the regex matching downstream.

Extraction and correction:

- Tesseract runs with `--oem 3 --psm 6` (default engine, one uniform block of text), which suits dense single-column documents.
- `standard_corrections()` fixes the digit/letter mix-ups these documents produce (`L0T` → `LOT`, `C0A` → `COA`, `CERT1FICATE`, `EXP1RATION`) and strips stray `|`, `_` and `+` left by table rules and scan noise. These are targeted substitutions; a general spellchecker would corrupt real part numbers.
- Regex pulls named fields (process run IDs, 8-digit lot numbers) out of the corrected text.

Localization and output:

- `image_to_data` gives a bounding box per word. Coordinates come from `main_body`, the same image OCR ran on, so the boxes line up with the text instead of being offset by the sidebar crop.
- Two overlays are drawn: every word above confidence 40, and only words matching `key_fields` (CERTIFICATE, LOT, BATCH, VENDOR, EXPIRATION, DATE, SIGNATURE, COA, MANUFACTURER).
- Field matching uses substrings, so `LOT:` and `LOT#` both match `LOT`.
- The output JSON maps each field to a list of hits (text, bbox, confidence), so a document with several dates keeps all of them.

## Digital-PDF extractor (`mini_pdfdataextraction_pymupdf.py`)

Digital PDFs already have a text layer, so there's no OCR. The hard part is that a regex matches against a string, while the answer you need is a location on the page.

- `build_positioned_text()` joins PyMuPDF's word tokens into one searchable string and records each word's character range and bounding box.
- `bbox_for_match()` takes a regex match's character span, finds every word that overlaps it, and returns the union of their boxes.

Together these let you write ordinary regex against readable text and still get page coordinates back.

- `FIELD_PATTERNS` holds one named pattern per field (vendor, document type, manufacturing date, expiration date, revision number, lot number), so adding a field means adding one dict entry.
- `is_real_date()` checks candidates against five date formats with `strptime`, which rejects part numbers that happen to be eight digits. `to_iso()` normalizes accepted dates to `YYYY-MM-DD`.
- `extract_table()` uses PyMuPDF's `find_tables()` and returns the bbox plus extracted rows.
- `draw_field_boxes()` renders the page at 2x zoom and draws labeled boxes, scaling coordinates by the same factor.

## Chat loop (`sample_chatbox.py`)

A minimal multi-turn chat over Gemini through LlamaIndex. It isolates the generation half of the pipeline so the LLM layer can be checked before retrieval is wired in. Both user and assistant turns go into a `ChatMessage` list so context carries across the conversation; `exit`, `quit` or `bye` ends it, and a try/except keeps one API error from killing the session.

The API key in the file is a placeholder. Set your own with an environment variable instead of pasting a real key:

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Running

Each file was written in Google Colab and has its own `!pip install` / `!apt-get` lines at the top. To run locally:

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

The scripts expect an input PDF in the working directory: `data.pdf` for the OCR pipeline and `sample-sdf-document.pdf` for the PyMuPDF extractor. No sample documents are committed, since the source documents are supplier-controlled.

## Notes

- The field patterns are tuned to one supplier's layout and need retuning for other vendors. That limit of regex extraction is part of why this project also has a RAG half.
- Notebooks are committed as `.py` exports. Colab notebooks store page images and widget state in their outputs, which bloats diffs and can leak document contents into version control.

## Stack

Python, PyMuPDF, Tesseract, PaddleOCR, EasyOCR, OpenCV, LlamaIndex, Gemini, NumPy, regex
