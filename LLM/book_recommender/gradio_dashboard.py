import html
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Setup

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
	raise RuntimeError("OPENROUTER_API_KEY not found in environment variables.")

DATA_DIR = Path("data")
BOOKS_CSV_PATH = DATA_DIR / "books_with_emotions.csv"
TAGGED_DESCRIPTION_PATH = DATA_DIR / "tagged_description.txt"
FALLBACK_COVER_PATH = str(DATA_DIR / "cover-not-found.jpg")

# Use a persistent Chroma directory so you don't rebuild embeddings every run.
CHROMA_DIR = DATA_DIR / "chroma_books_qwen3_embedding_8b"

# The emotions you used earlier; we'll visualise these if present in the CSV.
EMOTION_COLUMNS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# Reduce noisy logs
logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)


# Load data

books = pd.read_csv(BOOKS_CSV_PATH)

# Normalise ISBN to int where possible (keeps matching with retrieved IDs consistent).
books["isbn13"] = pd.to_numeric(books["isbn13"], errors="coerce").astype("Int64")

# More robust missing thumbnail handling: treat empty strings as missing too.
books["thumbnail"] = books["thumbnail"].replace("", np.nan)

# Large thumbnail URL (when available), fallback to local placeholder.
books["large_thumbnail"] = books["thumbnail"].apply(
	lambda url: f"{url}&fife=w800" if pd.notna(url) else FALLBACK_COVER_PATH
)


# Embeddings + Vector DB

embedding_model = OpenAIEmbeddings(
	model="qwen/qwen3-embedding-8b",
	openai_api_base="https://openrouter.ai/api/v1",
	openai_api_key=OPENROUTER_API_KEY,
	check_embedding_ctx_length=False,
)


def build_or_load_chroma() -> Chroma:
	"""
	Load an existing persisted Chroma DB if available; otherwise build it once.
	"""
	if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
		return Chroma(
			persist_directory=str(CHROMA_DIR),
			embedding_function=embedding_model,
		)

	raw_documents = TextLoader(str(TAGGED_DESCRIPTION_PATH), encoding="utf-8").load()
	splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
	documents = splitter.split_documents(raw_documents)

	db = Chroma.from_documents(
		documents=documents,
		embedding=embedding_model,
		persist_directory=str(CHROMA_DIR),
	)
	return db


db_books = build_or_load_chroma()


# Recommendation logic


def _format_authors(authors: Any) -> str:
	if not isinstance(authors, str) or not authors.strip():
		return "Unknown author"
	parts = [part.strip() for part in authors.split(";") if part.strip()]
	if len(parts) == 1:
		return parts[0]
	if len(parts) == 2:
		return f"{parts[0]} and {parts[1]}"
	return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _safe_int(value: Any) -> str:
	if pd.isna(value):
		return "—"
	try:
		return str(int(float(value)))
	except (ValueError, TypeError):
		return "—"


def _safe_float(value: Any, decimals: int = 2) -> str:
	if pd.isna(value):
		return "—"
	try:
		return f"{float(value):.{decimals}f}"
	except (ValueError, TypeError):
		return "—"


def _parse_isbn_from_doc_text(text: str) -> int | None:
	"""
	Your tagged_description lines appear to begin with an ISBN-like integer.
	We parse the first token safely.
	"""
	cleaned = text.strip().strip('"')
	if not cleaned:
		return None
	first_token = cleaned.split()[0]
	try:
		return int(first_token)
	except ValueError:
		return None


def retrieve_semantic_recommendations(
	query: str,
	category: str = "All",
	tone: str = "All",
	initial_top_k: int = 50,
	final_top_k: int = 18,
) -> pd.DataFrame:
	"""
	1) Retrieve initial_top_k candidate books from Chroma.
	2) Preserve the similarity order when mapping back to the DataFrame.
	3) Apply category filter and tone sorting, then take final_top_k.
	"""
	recs = db_books.similarity_search(query, k=initial_top_k)

	isbn_candidates: List[int] = []
	for rec in recs:
		isbn_value = _parse_isbn_from_doc_text(rec.page_content)
		if isbn_value is not None:
			isbn_candidates.append(isbn_value)

	if not isbn_candidates:
		return books.head(0)

	# Preserve similarity order: reindex by the candidate order.
	books_indexed = books.set_index("isbn13", drop=False)
	existing = books_indexed.loc[books_indexed.index.intersection(isbn_candidates)]
	ordered = existing.reindex([isbn for isbn in isbn_candidates if isbn in existing.index])

	filtered = ordered
	if category != "All":
		filtered = filtered[filtered["simple_categories"] == category]

	tone_to_column = {
		"Happy": "joy",
		"Surprising": "surprise",
		"Angry": "anger",
		"Suspenseful": "fear",
		"Sad": "sadness",
	}
	score_column = tone_to_column.get(tone)
	if score_column and score_column in filtered.columns:
		filtered = filtered.sort_values(by=score_column, ascending=False)

	return filtered.head(final_top_k)


# UI rendering helpers


def _emotion_bars_html(row: Dict[str, Any]) -> str:
	"""
	Render emotion scores as small bars if the columns exist.
	Values are expected in [0, 1]. If absent, return empty string.
	"""
	available = [col for col in EMOTION_COLUMNS if col in row and pd.notna(row[col])]
	if not available:
		return ""

	bars = []
	for col in available:
		try:
			value = float(row[col])
		except (ValueError, TypeError):
			continue
		value = max(0.0, min(1.0, value))
		width_pct = int(round(value * 100))

		label = html.escape(col)
		bars.append(
			f"""
            <div class="emo-row">
              <div class="emo-label">{label}</div>
              <div class="emo-track">
                <div class="emo-fill" style="width:{width_pct}%"></div>
              </div>
              <div class="emo-val">{value:.2f}</div>
            </div>
            """
		)

	if not bars:
		return ""

	return f"""
    <div class="emo-wrap">
      <div class="emo-title">Emotion scores</div>
      {"".join(bars)}
    </div>
    """


def _details_card_html(row: Dict[str, Any]) -> str:
	title = row.get("title_and_subtitle") or row.get("title") or "Untitled"
	authors = _format_authors(row.get("authors"))
	year = _safe_int(row.get("published_year"))
	pages = _safe_int(row.get("num_pages"))
	rating = _safe_float(row.get("average_rating"), decimals=2)
	category = row.get("simple_categories") or "—"
	desc = row.get("description")

	if not isinstance(desc, str) or not desc.strip():
		desc = "No description available."

	# Escape for safety
	title = html.escape(str(title))
	authors = html.escape(str(authors))
	category = html.escape(str(category))
	desc = html.escape(str(desc))

	emotions_section = _emotion_bars_html(row)

	return f"""
    <div class="book-card">
      <div class="book-title">{title}</div>
      <div class="book-authors">{authors}</div>

      <div class="book-meta">
        <span><b>Year</b>: {year}</span>
        <span><b>Pages</b>: {pages}</span>
        <span><b>Rating</b>: {rating}</span>
        <span><b>Category</b>: {category}</span>
      </div>

      <div class="book-desc">{desc}</div>
      {emotions_section}
    </div>
    """


def _normalise_gallery_index(index: Any, columns: int) -> int:
	"""
	Some Gradio builds provide evt.index as an int; others provide (row, col).
	"""
	if isinstance(index, (tuple, list)) and len(index) == 2:
		row, col = index
		return int(row) * columns + int(col)
	return int(index)


# Gradio callbacks

GALLERY_COLUMNS = 6
GALLERY_ROWS = 2
GALLERY_HEIGHT = 560


def recommend_and_prepare_state(
	query: str,
	category: str,
	tone: str,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]], str, str]:
	recommendations = retrieve_semantic_recommendations(
		query=query,
		category=category,
		tone=tone,
		initial_top_k=60,
		final_top_k=16,
	)

	rows = recommendations.to_dict(orient="records")

	gallery_items: List[Tuple[str, str]] = []
	for row in rows:
		short_title = row.get("title") or "Untitled"
		short_authors = _format_authors(row.get("authors"))
		short_year = _safe_int(row.get("published_year"))
		caption = f"{short_title}\n{short_authors} · {short_year}"

		gallery_items.append((row.get("large_thumbnail", FALLBACK_COVER_PATH), caption))

	if rows:
		first_cover = rows[0].get("large_thumbnail", FALLBACK_COVER_PATH)
		first_details = _details_card_html(rows[0])
	else:
		first_cover = FALLBACK_COVER_PATH
		first_details = "<div class='book-card'>No results.</div>"

	return gallery_items, rows, first_cover, first_details


def show_details(
	rows: List[Dict[str, Any]],
	evt: gr.SelectData,
) -> Tuple[str, str]:
	idx = _normalise_gallery_index(evt.index, columns=GALLERY_COLUMNS)
	if 0 <= idx < len(rows):
		row = rows[idx]
		cover = row.get("large_thumbnail", FALLBACK_COVER_PATH)
		return cover, _details_card_html(row)

	return FALLBACK_COVER_PATH, "<div class='book-card'>No details.</div>"


# UI

categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

CSS = f"""
/* Overall spacing polish */
.gradio-container {{
  max-width: 1400px !important;
}}

/* Gallery captions: small, readable, wrapped */
#book_gallery figcaption {{
  font-size: 13px;
  line-height: 1.25;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}

/* Selected cover styling */
#selected_cover img {{
  border-radius: 14px;
}}

/* Details panel: scrollable, proper typography */
#book_details {{
  max-height: {GALLERY_HEIGHT}px;
  overflow-y: auto;
}}

.book-card {{
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(0,0,0,0.25);
}}

.book-title {{
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 4px;
}}

.book-authors {{
  font-size: 15px;
  opacity: 0.92;
  margin-bottom: 10px;
}}

.book-meta {{
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  font-size: 13px;
  opacity: 0.90;
  margin-bottom: 12px;
}}

.book-desc {{
  font-size: 15px;
  line-height: 1.55;
  white-space: pre-wrap;
  margin-bottom: 14px;
}}

/* Emotion bars */
.emo-wrap {{
  margin-top: 8px;
  padding-top: 10px;
  border-top: 1px solid rgba(255,255,255,0.10);
}}

.emo-title {{
  font-size: 13px;
  font-weight: 700;
  opacity: 0.95;
  margin-bottom: 10px;
}}

.emo-row {{
  display: grid;
  grid-template-columns: 90px 1fr 46px;
  gap: 10px;
  align-items: center;
  margin-bottom: 8px;
}}

.emo-label {{
  font-size: 12px;
  opacity: 0.92;
}}

.emo-track {{
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.12);
  overflow: hidden;
}}

.emo-fill {{
  height: 100%;
  border-radius: 999px;
  background: rgba(255,255,255,0.55);
}}

.emo-val {{
  font-size: 12px;
  opacity: 0.92;
  text-align: right;
}}
"""

with gr.Blocks() as dashboard:
	gr.Markdown("# Semantic Book Recommender")

	with gr.Row():
		user_query = gr.Textbox(
			label="Please Enter a Description of a Book:",
			placeholder="e.g., A story about time travel",
		)
		category_dropdown = gr.Dropdown(
			choices=categories,
			label="Select a Category:",
			value="All",
		)
		tone_dropdown = gr.Dropdown(
			choices=tones,
			label="Select an Emotional Tone:",
			value="All",
		)
		submit_button = gr.Button("Find Recommendations")

	with gr.Row():
		with gr.Column(scale=2):
			gr.Markdown("## Recommendations")
			gallery = gr.Gallery(
				label="",
				columns=GALLERY_COLUMNS,
				rows=GALLERY_ROWS,
				height=GALLERY_HEIGHT,
				allow_preview=False,  # crucial: removes the cramped overlay UI
				object_fit="contain",
				elem_id="book_gallery",
			)

		with gr.Column(scale=1):
			gr.Markdown("## Selected book")
			selected_cover = gr.Image(
				label="",
				show_label=False,
				height=360,
				elem_id="selected_cover",
			)
			details_panel = gr.HTML(elem_id="book_details")

	rows_state = gr.State([])

	submit_button.click(
		fn=recommend_and_prepare_state,
		inputs=[user_query, category_dropdown, tone_dropdown],
		outputs=[gallery, rows_state, selected_cover, details_panel],
	)

	gallery.select(
		fn=show_details,
		inputs=[rows_state],
		outputs=[selected_cover, details_panel],
	)

if __name__ == "__main__":
	dashboard.launch(
		theme=gr.themes.Glass(),
		css=CSS,
	)
