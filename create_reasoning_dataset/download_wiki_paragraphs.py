#!/usr/bin/env python3
"""
Download and extract paragraphs from a specified Wikipedia dump (for Norwegian Wikipedia processing).
Non-text elements such as headings, lists, and media files are excluded.
The script outputs one JSON object per line in JSONL format:
{
  "url": <article URL>,
  "paragraph_number": <int>,
  "text": <paragraph text>
}

Example usage:
  python download_wiki_paragraphs.py --language no --output_file norwegian.jsonl --max_paragraphs 100 --minimum_words_paragraph 15

Dependencies:
  - requests (for downloading)
    pip install requests
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
    """
    Downloads the latest Wikipedia pages-articles dump for a specified language.
    
    Constructs the URL for the dump based on the language code,
    streams the content, and writes it to the provided output path.
    
    Parameters:
        language (str): Wikipedia language code (e.g., 'no' for Norwegian).
        output_path (str): File path where the downloaded dump will be saved.
    """
    url = f"https://dumps.wikimedia.org/{language}wiki/latest/{language}wiki-latest-pages-articles.xml.bz2"
    print(f"[INFO] Downloading Wikipedia dump from: {url}")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('Content-Length', 0))
        chunk_size = 8192  # 8 KB per chunk
        
        with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc='Downloading dump') as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"[INFO] Download complete: {output_path}")

def detect_namespace(bz2_file: str) -> str:
    """
    Detects and returns the XML namespace used in the Wikipedia dump.
    
    It parses the beginning of the bz2 file and extracts the namespace from the first encountered tag.
    
    Parameters:
        bz2_file (str): Path to the bz2 compressed Wikipedia dump.
        
    Returns:
        str: XML namespace string, including curly braces, e.g., "{http://www.mediawiki.org/xml/export-0.10/}".
             If detection fails, returns a fallback namespace.
    """
    with bz2.open(bz2_file, 'rb') as f:
        for event, elem in etree.iterparse(f, events=('start',)):
            if '}' in elem.tag:
                namespace = elem.tag.split('}')[0] + "}"
                print(f"[INFO] Detected namespace: {namespace}")
                return namespace
    # Fallback namespace if not detected
    return "{http://www.mediawiki.org/xml/export-0.10/}"

def is_valid_article(title: str) -> bool:
    """
    Determines if a Wikipedia page title corresponds to a valid article.
    
    Skips pages that are not main content, such as administrative or special pages.
    
    Parameters:
        title (str): The title of the Wikipedia page.
        
    Returns:
        bool: True if the article is valid; False otherwise.
    """
    excluded_prefixes = (
        "Wikipedia:", "Kategori:", "Fil:", "Mal:", "Hjelp:", "MediaWiki:", "Brukar:", "Diskusjon:"
    )
    if title.startswith(excluded_prefixes):
        return False
    if title == "Hovudside":  # Exclude the main page
        return False
    return True

def extract_paragraphs_from_page(wiki_text: str, min_words: int) -> list:
    """
    Extracts and filters paragraphs from raw Wikipedia wikitext.
    
    The function performs several cleaning steps:
      1. Removes templates.
      2. Removes file/image wikilinks.
      3. Removes HTML tags.
      4. Removes headings.
      
    After cleaning, it splits the text into paragraphs and filters out paragraphs that:
      - Start with a parenthesis.
      - Contain fewer than the specified minimum number of words.
      - Do not end with a valid punctuation mark (., !, ?, ,).
      - Contain undesirable patterns (e.g., ellipses, thumbnail markers).
    
    Parameters:
        wiki_text (str): The raw wikitext of the page.
        min_words (int): Minimum number of words required for a paragraph to be valid.
    
    Returns:
        list: A list of cleaned and validated paragraph strings.
    """
    parsed = mwparserfromhell.parse(wiki_text)

    # 1. Remove templates (process in reverse to avoid index issues)
    templates = list(parsed.ifilter_templates(recursive=True))
    for template in reversed(templates):
        parsed.remove(template)

    # 2. Remove File/Image links (also processed in reverse)
    file_links = [
        link for link in parsed.ifilter_wikilinks()
        if str(link.title).strip().startswith(("File:", "Image:"))
    ]
    for link in reversed(file_links):
        parsed.remove(link)

    # 3. Remove HTML tags
    html_tags = list(parsed.ifilter_tags())
    for tag in reversed(html_tags):
        parsed.remove(tag)

    # 4. Remove headings
    headings = list(parsed.ifilter_headings())
    for heading in reversed(headings):
        parsed.remove(heading)

    # Convert remaining content to plain text
    plain_text = parsed.strip_code(normalize=False, collapse=False)
    # Split text into paragraphs based on two or more newlines
    split_paragraphs = re.split(r'\n{2,}', plain_text)
    # Clean and normalize whitespace in paragraphs
    raw_paragraphs = [re.sub(r'\s+', ' ', p.strip()) for p in split_paragraphs if p.strip()]

    valid_endings = {'.', '!', '?', ','}
    paragraphs = []
    for p in raw_paragraphs:
        # Skip paragraphs that start with a parenthesis
        if p.startswith("("):
            continue
        
        words = p.split()
        if len(words) < min_words:
            continue
        
        # Ensure paragraph ends with a valid punctuation mark
        if p[-1] not in valid_endings:
            continue
        
        # Exclude paragraphs with unwanted patterns
        if "..." in p or "…" in p or "thumb|" in p:
            continue
        
        paragraphs.append(p)

    return paragraphs

def process_dump(language: str, bz2_file: str, output_file: str, max_paragraphs: int, min_words: int):
    """
    Processes the compressed Wikipedia dump to extract paragraphs and write them to a JSONL file.
    
    The function iterates through pages in the XML dump, filters valid articles,
    extracts paragraphs from the wikitext, and writes each paragraph along with its metadata.
    Processing stops once the maximum number of paragraphs is reached.
    
    Parameters:
        language (str): Wikipedia language code.
        bz2_file (str): Path to the bz2 compressed Wikipedia dump.
        output_file (str): File path for the output JSONL file.
        max_paragraphs (int): Maximum number of paragraphs to extract.
        min_words (int): Minimum number of words required for each paragraph.
    """
    print(f"[INFO] Processing dump: {bz2_file}")
    print(f"[INFO] Writing JSONL paragraphs to: {output_file}")

    namespace = detect_namespace(bz2_file)
    total_paragraphs = 0

    with bz2.open(bz2_file, 'rb') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        # Set up an iterative parser for <page> elements using the detected namespace.
        context = etree.iterparse(f, events=('end',), tag=f'{namespace}page')

        with tqdm(desc='Parsing pages', unit=' pages') as pbar:
            for event, elem in context:
                # Retrieve title and revision elements
                title_el = elem.find(f'{namespace}title')
                revision_el = elem.find(f'{namespace}revision')

                if title_el is not None and revision_el is not None:
                    title = title_el.text if title_el.text else ""

                    # Skip non-article pages
                    if not is_valid_article(title):
                        elem.clear()
                        continue

                    text_el = revision_el.find(f'{namespace}text')
                    if text_el is not None and text_el.text:
                        paragraphs = extract_paragraphs_from_page(text_el.text, min_words)

                        # Construct the article URL by replacing spaces with underscores
                        url_title = title.replace(' ', '_')
                        page_url = f"https://{language}.wikipedia.org/wiki/{url_title}"

                        # Write each valid paragraph to the output file
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

                    pbar.update(1)  # Update progress for each processed page

                # Clear memory for the processed element
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

    print(f"[INFO] Total paragraphs extracted: {total_paragraphs}")

def main():
    """
    Main function to handle argument parsing and orchestrate the download and processing
    of the Wikipedia dump.
    
    Steps:
      1. Parse command-line arguments.
      2. Check for the existence of the Wikipedia dump file; download it if missing.
      3. Process the dump to extract paragraphs and write them in JSONL format.
      4. Log progress and completion.
    """
    parser = argparse.ArgumentParser(
        description="Download a Wikipedia dump and extract paragraphs into a JSONL file."
    )
    parser.add_argument("--language", type=str, required=True,
                        help="Wikipedia language code (e.g., en, no, de, fr).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path for the resulting JSON lines.")
    parser.add_argument("--temp_dump_file", type=str, default="temp_wiki_dump.xml.bz2",
                        help="Path to store or read the downloaded Wikipedia dump (bz2).")
    parser.add_argument("--max_paragraphs", type=int, default=10_000_000,
                        help="Maximum number of paragraphs to extract (default: 10M).")
    parser.add_argument("--minimum_words_paragraph", type=int, default=15,
                        help="Minimum words per paragraph (default: 15).")

    args = parser.parse_args()

    # Download Wikipedia dump if it does not exist or is empty
    if not os.path.exists(args.temp_dump_file) or os.path.getsize(args.temp_dump_file) == 0:
        print(f"[INFO] Wikipedia dump not found. Downloading: {args.temp_dump_file}")
        download_wiki_dump(args.language, args.temp_dump_file)

    # Process the downloaded dump and extract paragraphs
    process_dump(
        args.language,
        args.temp_dump_file,
        args.output_file,
        args.max_paragraphs,
        args.minimum_words_paragraph
    )

    print("[INFO] Done.")

if __name__ == "__main__":
    main()

