"""
E-JUST Cloud Scraper & PDF Extractor
------------------------------------
Replaces the Colab notebook with a clean, reproducible script designed to run
in GitHub Actions (or any Linux/Windows runner).

Improvements over the original notebook:
1. Fixes title truncation bug ("E-JUST" -> "E").
2. Converts HTML to structured Markdown (preserving headings, lists, and tables).
3. Deduplicates pages by canonical URL AND SHA-256 content hash.
4. Extracts text page-by-page from downloaded E-JUST PDFs using PyMuPDF.
5. Writes clean .md files with YAML frontmatter into knowledge/scraped/.
"""

import argparse
import hashlib
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import pymupdf
import requests
import urllib3
from bs4 import BeautifulSoup, Tag

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ejust_scraper")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "knowledge" / "scraped"
PDF_TMP_DIR = ROOT / ".cache_pdfs"

SEED_URLS = [
    "https://www.ejust.edu.eg/",
    "https://www.ejust.edu.eg/academics",
    "https://www.ejust.edu.eg/research",
    "https://www.ejust.edu.eg/news",
    "https://www.ejust.edu.eg/events",
    "https://www.ejust.edu.eg/library",
    "https://www.ejust.edu.eg/csit",
    "https://www.ejust.edu.eg/bas",
    "https://www.ejust.edu.eg/pharmd",
    "https://www.ejust.edu.eg/art-and-design",
    "https://csit.ejust.edu.eg/",
    "https://academic-profile.ejust.edu.eg/",
    "https://ugad.ejust.edu.eg/",
]

SKIP_PATTERNS = [
    r"\.(jpg|jpeg|png|gif|svg|ico|mp4|mp3|avi|zip|rar|exe|dmg|css|js|woff2?)$",
    r"wp-login",
    r"wp-admin",
    r"xmlrpc",
    r"^#",
    r"javascript:",
    r"mailto:",
    r"tel:",
    r"/feed/",
    r"/rss/",
    r"/tag/",
    r"/page/\d+/",
    r"\?s=",
    r"replytocom=",
]


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url.strip())
    # Some pages contain malformed hrefs like "/https://library.ejust.edu.eg"
    url = re.sub(r"^(https?://[^/]+)?/(https?://)", r"\2", url)
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc == "ejust.edu.eg":
        netloc = "www.ejust.edu.eg"
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{netloc}{path}" + (f"?{parsed.query}" if parsed.query else "")


def is_allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc.lower().endswith("ejust.edu.eg")
    except Exception:
        return False


def should_skip(url: str) -> bool:
    return any(re.search(pat, url, re.IGNORECASE) for pat in SKIP_PATTERNS)


def clean_title(raw_title: str) -> str:
    if not raw_title:
        return "E-JUST Information"
    # Strip trailing site suffix like " | E-JUST" or " – Egypt-Japan University..."
    # without breaking "E-JUST" itself.
    cleaned = re.sub(
        r"\s*[|–—]\s*(E-?JUST|Egypt-Japan University.*)?$",
        "",
        raw_title.strip(),
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or raw_title.strip()


def infer_label(url: str, title: str) -> str:
    hay = f"{url} {title}".lower()
    mapping = [
        (["admission", "apply", "enroll", "requirement", "ugad"], "admission"),
        (["fee", "tuition", "payment", "cost"], "fees"),
        (["scholarship", "grant", "fellowship", "ticad"], "scholarship"),
        (["program", "degree", "major", "specialization"], "program"),
        (["course", "curriculum", "syllabus"], "courses"),
        (["faculty", "staff", "professor", "instructor", "academic-profile"], "staff"),
        (["research", "publication", "lab", "project"], "research"),
        (["news", "announcement", "press"], "news"),
        (["event", "seminar", "workshop", "conference"], "events"),
        (["about", "overview", "vision", "mission"], "about"),
        (["contact", "location", "address"], "contact"),
        (["student", "campus", "club", "dorm"], "student_life"),
        (["graduate", "msc", "phd", "postgraduate"], "postgraduate"),
        (["undergraduate", "bachelor"], "undergraduate"),
        (["library", "journal", "repository"], "library"),
    ]
    for keywords, label in mapping:
        if any(k in hay for k in keywords):
            return label
    return "general"


def infer_faculty(url: str) -> str:
    u = url.lower()
    if "csit" in u:
        return "CSIT"
    if "art" in u or "design" in u:
        return "Art & Design"
    if "pharm" in u:
        return "Pharmacy"
    if "bas" in u:
        return "Basic & Applied Science"
    if "engineering" in u:
        return "Engineering"
    return "EJUST General"


def table_to_markdown(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).replace("|", "\\|")
            for cell in tr.find_all(["th", "td"])
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    norm_rows = [r + [""] * (col_count - len(r)) for r in rows]
    lines = [
        "| " + " | ".join(norm_rows[0]) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in norm_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n" + "\n".join(lines) + "\n"


def html_to_markdown(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(
        ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form", "button", "svg"]
    ):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|post", re.I))
        or soup.find(class_=re.compile(r"content|main|post|entry", re.I))
        or soup.body
    )
    if not main:
        return ""

    # Convert tables first so their text isn't duplicated
    for tbl in main.find_all("table"):
        md_tbl = table_to_markdown(tbl)
        tbl.replace_with(soup.new_string(f"\n{md_tbl}\n"))

    blocks = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"], recursive=True):
        # Skip nested paragraphs inside list items already handled
        if el.name == "p" and el.find_parent("li"):
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if el.name == "h1":
            blocks.append(f"# {text}")
        elif el.name == "h2":
            blocks.append(f"## {text}")
        elif el.name in ("h3", "h4"):
            blocks.append(f"### {text}")
        elif el.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    md = "\n\n".join(blocks)
    if len(md.split()) < 40:
        # Fallback to plain text if page doesn't use standard p/h tags
        md = re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip()
    return md


def detect_lang(soup: BeautifulSoup, text: str) -> str:
    html_tag = soup.find("html")
    html_lang = (html_tag.get("lang", "") if html_tag else "").lower()
    if "ar" in html_lang:
        return "ar"
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text[:500]))
    return "ar" if arabic_chars > 30 else "en"


def write_markdown_doc(
    doc_id: str,
    title: str,
    url: str,
    faculty: str,
    label: str,
    lang: str,
    source_type: str,
    body: str,
) -> None:
    safe_title = title.replace("\n", " ").replace(":", " -").strip()
    frontmatter = (
        "---\n"
        f"title: {safe_title}\n"
        f"url: {url}\n"
        f"faculty: {faculty}\n"
        f"label: {label}\n"
        f"lang: {lang}\n"
        f"source_type: {source_type}\n"
        "---\n\n"
    )
    (OUT_DIR / f"{doc_id}.md").write_text(frontmatter + body.strip() + "\n", encoding="utf-8")


def extract_pdf_markdown(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    pages_md = []
    for i, page in enumerate(doc, start=1):
        txt = re.sub(r"\n{3,}", "\n\n", page.get_text("text")).strip()
        if len(txt.split()) >= 15:
            pages_md.append(f"## Page {i}\n\n{txt}")
    return "\n\n".join(pages_md)


def safe_get(session: requests.Session, url: str, timeout: int = 20, stream: bool = False) -> requests.Response:
    try:
        return session.get(url, timeout=timeout, allow_redirects=True, stream=stream)
    except requests.exceptions.SSLError:
        # E-JUST subdomains sometimes omit intermediate SSL certs on Linux
        return session.get(url, timeout=timeout, allow_redirects=True, stream=stream, verify=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=600)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--skip-pdfs", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous scraped output so stale pages don't linger
    for old_file in OUT_DIR.glob("*.md"):
        old_file.unlink()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; EJUST-RAG-Bot/2.0)",
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        }
    )

    visited: set[str] = set()
    content_hashes: set[str] = set()
    queue: deque[str] = deque(normalize_url(u) for u in SEED_URLS)
    queued: set[str] = set(queue)
    pdf_candidates: dict[str, str] = {}
    pages_saved = 0

    while queue and len(visited) < args.max_pages:
        url = queue.popleft()
        queued.discard(url)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = safe_get(session, url, timeout=20)
            if resp.status_code != 200:
                continue
            if "html" not in resp.headers.get("Content-Type", "").lower():
                continue
        except Exception as exc:
            log.warning("Fetch failed %s: %s", url, exc)
            continue

        # Parse raw bytes so BeautifulSoup honours the page's own <meta charset>.
        # (resp.text falls back to ISO-8859-1 and garbles Arabic pages.)
        soup = BeautifulSoup(resp.content, "lxml")

        # Discover links & PDFs before decomposing nav/footer
        for a in soup.find_all("a", href=True):
            abs_url = normalize_url(urljoin(resp.url, a["href"]))
            if ".pdf" in abs_url.lower() and is_allowed(abs_url):
                if abs_url not in pdf_candidates:
                    pdf_candidates[abs_url] = a.get_text(strip=True) or Path(urlparse(abs_url).path).stem
                continue
            if is_allowed(abs_url) and not should_skip(abs_url):
                if abs_url not in visited and abs_url not in queued:
                    queue.append(abs_url)
                    queued.add(abs_url)

        raw_title = soup.title.get_text(strip=True) if soup.title else ""
        title = clean_title(raw_title)
        md_body = html_to_markdown(soup)

        if len(md_body.split()) < 40:
            time.sleep(args.delay)
            continue

        norm_hash = hashlib.sha256(re.sub(r"\s+", " ", md_body).encode("utf-8")).hexdigest()
        if norm_hash in content_hashes:
            time.sleep(args.delay)
            continue
        content_hashes.add(norm_hash)

        doc_id = f"web_{hashlib.md5(resp.url.encode()).hexdigest()[:12]}"
        write_markdown_doc(
            doc_id=doc_id,
            title=title,
            url=resp.url,
            faculty=infer_faculty(resp.url),
            label=infer_label(resp.url, title),
            lang=detect_lang(soup, md_body),
            source_type="webpage",
            body=md_body,
        )
        pages_saved += 1
        if pages_saved % 25 == 0:
            log.info("Saved %d unique pages (visited=%d, queue=%d, pdfs=%d)", pages_saved, len(visited), len(queue), len(pdf_candidates))

        time.sleep(args.delay)

    log.info("Web crawl finished: %d unique pages saved, %d PDFs discovered", pages_saved, len(pdf_candidates))

    if args.skip_pdfs or not pdf_candidates:
        return

    pdf_hashes: set[str] = set()
    pdfs_saved = 0
    for pdf_url, pdf_title in pdf_candidates.items():
        uid = hashlib.md5(pdf_url.encode()).hexdigest()[:12]
        pdf_path = PDF_TMP_DIR / f"{uid}.pdf"
        try:
            r = safe_get(session, pdf_url, timeout=25, stream=True)
            if r.status_code != 200:
                continue
            data = r.content
            if len(data) > 30 * 1024 * 1024:
                continue
            file_hash = hashlib.sha256(data).hexdigest()
            if file_hash in pdf_hashes:
                continue
            pdf_hashes.add(file_hash)
            pdf_path.write_bytes(data)

            pdf_md = extract_pdf_markdown(pdf_path)
            if len(pdf_md.split()) < 30:
                continue

            write_markdown_doc(
                doc_id=f"pdf_{uid}",
                title=pdf_title.replace("-", " ").replace("_", " ").strip() or "E-JUST Document",
                url=pdf_url,
                faculty=infer_faculty(pdf_url),
                label=infer_label(pdf_url, pdf_title),
                lang="ar" if len(re.findall(r"[\u0600-\u06FF]", pdf_md[:500])) > 30 else "en",
                source_type="pdf",
                body=pdf_md,
            )
            pdfs_saved += 1
        except Exception as exc:
            log.warning("PDF failed %s: %s", pdf_url, exc)

    log.info("PDF extraction finished: %d PDFs converted to Markdown", pdfs_saved)


if __name__ == "__main__":
    main()
