#!/usr/bin/env python3
"""
Optimized script to download and extract paragraphs from a Wikipedia dump using parallel processing.
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
from multiprocessing import Pool, cpu_count

# Precompile regex patterns for efficiency
PARAGRAPH_SPLIT_RE = re.compile(r'\n{2,}')
WHITESPACE_RE = re.compile(r'\s+')
ELLIPSIS_RE = re.compile(r'\.\.\.|…')
THUMB_RE = re.compile(r'thumb\|', re.IGNORECASE)
PAREN_START_RE = re.compile(r'^\(')
VALID_ENDINGS = {'.', '!', '?', ','}

def download_wiki_dump(language: str, output_path: str):
    """Download Wikipedia dump for the specified language."""
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

def detect_namespace(bz2_file: str) -> str:
    """Detect XML namespace from the dump file."""
    with bz2.open(bz2_file, 'rb') as f:
        for event, elem in etree.iterparse(f, events=('start',)):
            if '}' in elem.tag:
                namespace = elem.tag.split('}')[0] + "}"
                print(f"[INFO] Detected namespace: {namespace}")
                return namespace
    return "{http://www.mediawiki.org/xml/export-0.10/}"

def is_valid_article(title: str) -> bool:
    """Check if a page title corresponds to a valid article."""
    excluded_prefixes = (
        "Wikipedia:", "Kategori:", "Fil:", "Mal:", "Hjelp:", "MediaWiki:", "Brukar:", "Diskusjon:"
    )
    return not title.startswith(excluded_prefixes) and title != "Hovudside"

def extract_paragraphs_from_page(wiki_text: str, min_words: int) -> list:
    """Extract and filter paragraphs from wikitext."""
    parsed = mwparserfromhell.parse(wiki_text)

    # Remove unwanted elements in reverse to avoid index issues
    for template in reversed(list(parsed.ifilter_templates(recursive=True))):
        parsed.remove(template)
    for link in reversed([link for link in parsed.ifilter_wikilinks() if str(link.title).strip().startswith(("File:", "Image:"))]):
        parsed.remove(link)
    for tag in reversed(list(parsed.ifilter_tags())):
        parsed.remove(tag)
    for heading in reversed(list(parsed.ifilter_headings())):
        parsed.remove(heading)

    plain_text = parsed.strip_code(normalize=False, collapse=False)
    split_paragraphs = PARAGRAPH_SPLIT_RE.split(plain_text)
    raw_paragraphs = [WHITESPACE_RE.sub(' ', p.strip()) for p in split_paragraphs if p.strip()]

    paragraphs = []
    for p in raw_paragraphs:
        p = p.replace(", (),", "").replace("() ", "")
        if not p or not p[0].isupper() or PAREN_START_RE.match(p):
            continue
        words = p.split()
        if len(words) < min_words or p[-1] not in VALID_ENDINGS:
            continue
        if ELLIPSIS_RE.search(p) or THUMB_RE.search(p):
            continue
        paragraphs.append(p)
    return paragraphs

def process_page(args):
    """Process a single page's text into paragraphs (for parallel processing)."""
    text, min_words, page_url = args
    paragraphs = extract_paragraphs_from_page(text, min_words)
    return [{
        "url": page_url,
        "paragraph_number": idx + 1,
        "text": p
    } for idx, p in enumerate(paragraphs)]

def process_dump(language: str, bz2_file: str, output_file: str, max_paragraphs: int, min_words: int):
    """
    Process the dump using parallel workers to extract paragraphs.

    Updated to use specific element checks instead of generic truth checks.
    """
    print(f"[INFO] Processing dump: {bz2_file}")
    namespace = detect_namespace(bz2_file)

    stop_processing = False
    total_paragraphs = 0

    with bz2.open(bz2_file, 'rb') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        context = etree.iterparse(f, events=('end',), tag=f'{namespace}page')
        pool = Pool(processes=cpu_count())

        def process_callback(results):
            nonlocal stop_processing, total_paragraphs
            if stop_processing:
                return
            for record in results:
                if total_paragraphs >= max_paragraphs:
                    stop_processing = True
                    break
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_paragraphs += 1
                if total_paragraphs >= max_paragraphs:
                    stop_processing = True
                    break

        try:
            with tqdm(desc='Parsing pages', unit=' pages') as pbar:
                for _, elem in context:
                    if stop_processing:
                        break

                    title_el = elem.find(f'{namespace}title')
                    revision_el = elem.find(f'{namespace}revision')

                    # Use explicit None checks
                    if title_el is None or revision_el is None:
                        elem.clear()
                        continue

                    title = title_el.text or ""
                    if not is_valid_article(title):
                        elem.clear()
                        continue

                    text_el = revision_el.find(f'{namespace}text')
                    if text_el is None or text_el.text is None:
                        elem.clear()
                        continue

                    page_url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"

                    pool.apply_async(
                        process_page,
                        args=((text_el.text, min_words, page_url),),
                        callback=process_callback
                    )

                    pbar.update(1)
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

        finally:
            pool.close()
            pool.join()

    print(f"[INFO] Total paragraphs extracted: {total_paragraphs}")

def main():
    """Main function to handle arguments and orchestrate processing."""
    parser = argparse.ArgumentParser(description="Download and process a Wikipedia dump into paragraphs.")
    parser.add_argument("--language", required=True, help="Wikipedia language code.")
    parser.add_argument("--output_file", required=True, help="Output JSONL file path.")
    parser.add_argument("--temp_dump_file", default="temp_wiki_dump.xml.bz2", help="Temporary dump file path.")
    parser.add_argument("--max_paragraphs", type=int, default=10_000_000, help="Maximum paragraphs to extract.")
    parser.add_argument("--minimum_words_paragraph", type=int, default=15, help="Minimum words per paragraph.")

    args = parser.parse_args()

    if not os.path.exists(args.temp_dump_file) or os.path.getsize(args.temp_dump_file) == 0:
        download_wiki_dump(args.language, args.temp_dump_file)

    process_dump(
        args.language,
        args.temp_dump_file,
        args.output_file,
        args.max_paragraphs,
        args.minimum_words_paragraph
    )

if __name__ == "__main__":
    main()

