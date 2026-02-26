from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path("output/pdf/ilga_graph_app_summary_one_page.pdf")
PNG_PREVIEW = Path("tmp/pdfs/render/ilga_graph_app_summary_one_page.png")

W, H = 1275, 1650  # 8.5x11 at 150 dpi
MARGIN_X = 90
TOP = 80

img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    font_heading = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    font_body = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
except Exception:
    font_title = ImageFont.load_default()
    font_heading = ImageFont.load_default()
    font_body = ImageFont.load_default()

content: list[tuple[str, str]] = [
    ("title", "ILGA Graph App - One-Page Summary"),
    ("space", ""),
    ("heading", "What It Is"),
    ("body", "ILGA Graph (branded as 'The Land of Kei') is a FastAPI web app focused on"),
    ("body", "Illinois advocacy for legal kei vehicle registration, plus a GraphQL API."),
    ("space", ""),
    ("heading", "Who It's For"),
    ("body", "Primary persona: Illinois constituents and campaign organizers who need"),
    ("body", "to quickly identify legislators, perform outreach, and track engagement."),
    ("space", ""),
    ("heading", "What It Does"),
    ("body", "- ZIP lookup finds your senator, representative, and a power-broker target."),
    ("body", "- Guided call/email drawer provides prewritten outreach scripts and templates."),
    ("body", "- Outreach progress tracking supports signed-in users and signed-out fallback."),
    ("body", "- Intelligence dashboards show bill/member deep-dives, anomalies, and trends."),
    ("body", "- Power Map page visualizes influence and co-sponsorship relationships."),
    ("body", "- GraphQL endpoint exposes members, bills, committees, votes, slips, search."),
    ("space", ""),
    ("heading", "How It Works (Repo Evidence)"),
    ("body", "- Ingestion: scrapers + ETL load ILGA data (members, bills, votes, slips)."),
    ("body", "- Startup: lifespan loads cache/scrape path, computes analytics and graph data."),
    ("body", "- Runtime state: app_state holds members/bills/committees plus analytics."),
    ("body", "- Serving: FastAPI routers render Jinja pages; Strawberry GraphQL at /graphql."),
    ("body", "- Persistence: async SQLAlchemy + SQLite (data/ilga.db), Alembic migrations."),
    ("body", "- Outputs: optional ML artifacts in processed/ and Obsidian vault export."),
    ("space", ""),
    ("heading", "How To Run (Minimal)"),
    ("body", "1. Install deps: make install"),
    ("body", "2. Start app: make dev"),
    ("body", "3. Open: http://127.0.0.1:8000"),
    ("body", "4. Optional fresh legislative data: make scrape"),
    ("space", ""),
    ("body", "Not found in repo: a single canonical production hosting topology diagram."),
]

y = TOP
for kind, text in content:
    if kind == "title":
        font = font_title
        fill = "black"
        y += 0
    elif kind == "heading":
        font = font_heading
        fill = "black"
    elif kind == "body":
        font = font_body
        fill = "black"
    else:
        y += 10
        continue

    draw.text((MARGIN_X, y), text, fill=fill, font=font)
    bbox = draw.textbbox((MARGIN_X, y), text, font=font)
    line_h = bbox[3] - bbox[1]
    y += line_h + 8

if y > H - 70:
    raise SystemExit("Layout overflowed one page; reduce content.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
PNG_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
img.save(PNG_PREVIEW, "PNG")
img.save(OUTPUT, "PDF", resolution=150.0)
print(OUTPUT)
print(PNG_PREVIEW)
