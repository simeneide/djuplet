#!/usr/bin/env python3
"""
Download and extract paragraphs from a specified Wikipedia dump.
Headings, lists, and non-text elements are excluded.
Outputs one JSON object per line in JSONL format:
{
  "url": <article URL>,
  "paragraph_number": <int>,
  "text": <paragraph text>
}

Example usage:
  python wiki_downloader.py --language nn --output_file nynorsk.jsonl --max_paragraphs 100 --minimum_words_paragraph 15

Dependencies:
  - requests (for downloading)
  - mwparserfromhell (for parsing wikitext)
     pip install mwparserfromhell
  - lxml (for streaming through XML)
     pip install lxml
  - tqdm (for progress bars)
     pip install tqdm

"""

import os
import bz2
import argparse
import json
import requests
import re
from lxml import etree
import mwparserfromhell
from tqdm import tqdm

def download_wiki_dump(language: str, output_path: str):
    """Download the latest Wikipedia pages-articles dump for a given language."""
    url = f"https://dumps.wikimedia.org/{language}wiki/latest/{language}wiki-latest-pages-articles.xml.bz2"
    print(f"[INFO] Downloading Wikipedia dump from: {url}")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('Content-Length', 0))
        chunk_size = 8192
        
        with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc='Downloading dump') as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"[INFO] Download complete: {output_path}")

def detect_namespace(bz2_file: str):
    """Detects the XML namespace in the Wikipedia dump."""
    with bz2.open(bz2_file, 'rb') as f:
        for event, elem in etree.iterparse(f, events=('start',)):
            if '}' in elem.tag:
                namespace = elem.tag.split('}')[0] + "}"
                print(f"[INFO] Detected namespace: {namespace}")
                return namespace
    return "{http://www.mediawiki.org/xml/export-0.10/}"  # Fallback

def is_valid_article(title: str):
    """Returns True if the page is a valid Wikipedia article."""
    if title.startswith(("Wikipedia:", "Kategori:", "Fil:", "Mal:", "Hjelp:", "MediaWiki:", "Brukar:", "Diskusjon:")):
        return False
    if title == "Hovudside":  # Skip main page
        return False
    return True

def extract_paragraphs_from_page(wiki_text: str, min_words: int):
    """
    Parses wikitext while removing:
      - Templates (e.g., {{Infobox ... }})
      - Lists (e.g., "* item", "# numbered item")
      - Very short paragraphs (< min_words)
      - Paragraphs not ending in proper punctuation (. , ! ?)
      - Paragraphs containing "..." or "…"
    """
    parsed = mwparserfromhell.parse(wiki_text)

    # Remove templates
    for template in parsed.ifilter_templates():
        wiki_text = wiki_text.replace(str(template), "")

    parsed = mwparserfromhell.parse(wiki_text)  # Re-parse without templates

    # Get plain text (strip unnecessary wiki formatting)
    plain_text = parsed.strip_code(normalize=True, collapse=True)

    # Split text into paragraphs (ensuring each paragraph is a single line)
    raw_paragraphs = [re.sub(r'\s+', ' ', p.strip()) for p in plain_text.split("\n\n") if p.strip()]

    # Define valid ending punctuation for paragraphs
    valid_endings = {'.', '!', '?', ','}

    # Filter paragraphs
    paragraphs = []
    for paragraph in raw_paragraphs:
        words = paragraph.split()
        if len(words) < min_words:  # Remove very short paragraphs
            continue
        if paragraph[-1] not in valid_endings:  # Remove paragraphs without proper ending
            continue
        if "..." in paragraph or "…" in paragraph:  # Remove paragraphs with "..." or "…"
            continue
        paragraphs.append(paragraph)

    return paragraphs

def process_dump(language: str, bz2_file: str, output_file: str, max_paragraphs: int, min_words: int):
    """
    Streams through the Wikipedia dump XML (bz2_file), extracts paragraphs,
    and writes them line-by-line in JSONL format to output_file.
    Stops processing when max_paragraphs is reached.
    """
    print(f"[INFO] Processing dump: {bz2_file}")
    print(f"[INFO] Writing JSONL paragraphs to: {output_file}")

    namespace = detect_namespace(bz2_file)
    total_paragraphs = 0

    with bz2.open(bz2_file, 'rb') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        context = etree.iterparse(f, events=('end',), tag=f'{namespace}page')

        with tqdm(desc='Parsing pages', unit=' pages') as pbar:
            for event, elem in context:
                title_el = elem.find(f'{namespace}title')
                revision_el = elem.find(f'{namespace}revision')

                if title_el is not None and revision_el is not None:
                    title = title_el.text if title_el.text else ""

                    # Skip non-articles
                    if not is_valid_article(title):
                        elem.clear()
                        continue

                    text_el = revision_el.find(f'{namespace}text')
                    if text_el is not None and text_el.text:
                        paragraphs = extract_paragraphs_from_page(text_el.text, min_words)

                        # Construct Wikipedia URL
                        url_title = title.replace(' ', '_')
                        page_url = f"https://{language}.wikipedia.org/wiki/{url_title}"

                        for idx, p in enumerate(paragraphs):
                            record = {
                                "url": page_url,
                                "paragraph_number": idx + 1,
                                "text": p
                            }
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_paragraphs += 1

                            if total_paragraphs >= max_paragraphs:
                                print(f"[INFO] Reached max_paragraphs limit ({max_paragraphs}). Stopping.")
                                return

                    pbar.update(1)  # Update progress bar

                # Clear memory
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

    print(f"[INFO] Total paragraphs extracted: {total_paragraphs}")

def main():
    parser = argparse.ArgumentParser(
        description="Download a Wikipedia dump for a given language and extract paragraphs to JSON lines."
    )
    parser.add_argument("--language", type=str, required=True, help="Wikipedia language code (e.g., en, de, fr, no).")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path for the resulting JSON lines.")
    parser.add_argument("--temp_dump_file", type=str, default="temp_wiki_dump.xml.bz2",
                        help="Path to store or read the downloaded Wikipedia dump (bz2).")
    parser.add_argument("--max_paragraphs", type=int, default=10_000_000,
                        help="Maximum number of paragraphs to extract (default: 10M).")
    parser.add_argument("--minimum_words_paragraph", type=int, default=15,
                        help="Minimum words per paragraph (default: 15).")

    args = parser.parse_args()

    # 🔹 **Check if the Wikipedia dump exists; if not, download it**
    if not os.path.exists(args.temp_dump_file) or os.path.getsize(args.temp_dump_file) == 0:
        print(f"[INFO] Wikipedia dump not found. Downloading: {args.temp_dump_file}")
        download_wiki_dump(args.language, args.temp_dump_file)

    # 🔹 **Process the Wikipedia dump**
    process_dump(args.language, args.temp_dump_file, args.output_file, args.max_paragraphs, args.minimum_words_paragraph)

    print("[INFO] Done.")

if __name__ == "__main__":
    main()

