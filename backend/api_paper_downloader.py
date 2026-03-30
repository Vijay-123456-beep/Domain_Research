#!/usr/bin/env python3
"""
=============================================================================
 Biomass Supercapacitor — Federated Academic API Downloader
=============================================================================
 
 Aggregates search results across 6 major academic search engines:
 1. Semantic Scholar (API)
 2. Google Scholar (HTML Scraping - Fragile to IP Blocks)
 3. CrossRef (API)
 4. CORE API (API)
 5. PubMed Central / Europe PMC (API)
 6. arXiv (API)
 
 This script applies the user's strict keyword logic locally and saves the
 open-access PDFs into the PDFs directory, logging metadata into a CSV.
=============================================================================
"""

import os
import time
import re
import csv
import json
import logging
import urllib.parse
import argparse
import hashlib
from bs4 import BeautifulSoup
from schema_loader import load_domain_schema

try:
    import requests
    import pandas as pd
    import fitz # PyMuPDF
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Please install: pip install requests pandas bs4 pymupdf")
    exit(1)

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Defaults - will be overridden by --workspace)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(os.path.dirname(BASE_DIR), "PDFs")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")
LOG_FILE = os.path.join(BASE_DIR, "download_log.txt")

def setup_paths(workspace=None):
    global PDF_DIR, METADATA_FILE, LOG_FILE
    if workspace:
        # workspace is /home/cpad/Domain_Research/sessions/<task_id>
        # project_root is /home/cpad/Domain_Research
        project_root = os.path.dirname(os.path.dirname(workspace))
        task_id = os.path.basename(workspace)
        
        PDF_DIR = os.path.join(project_root, "PDFs", task_id)
        METADATA_FILE = os.path.join(workspace, "metadata.csv")
        LOG_FILE = os.path.join(workspace, "download_log.txt")
    
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    
    # Configure logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

# Logging is now set up in setup_paths()

def log(msg, level="info"):
    if level == "info": logging.info(msg)
    elif level == "error": logging.error(msg)
    elif level == "warning": logging.warning(msg)

# ---------------------------------------------------------
# FILTERING LOGIC
# ---------------------------------------------------------
def calculate_relevance(title, abstract, primary_kw, secondary_kw, original_query=None, domain_units=None):
    """
    Returns a float from 0.0 - 1.0 indicating paper relevance.
    STRICT REQUIREMENT: MUST hit at least 1 primary keyword match.
    Formula: 0.25*Query + 0.25*PrimaryKW + 0.25*Tech + 0.15*Num + 0.10*Domain
    """
    text = f"{title or ''} {abstract or ''}".lower()
    
    query_match = 0.0
    if original_query:
        stop_terms = {'machine', 'learning', 'model', 'method', 'using', 'study', 'analysis', 'system', 'data'}
        query_terms = [w.lower() for w in original_query.split() if len(w) > 3 and w.lower() not in stop_terms]
        if not query_terms:
            query_terms = [w.lower() for w in original_query.split() if len(w) > 3]
            
        if any(term in text for term in query_terms):
            query_match = 1.0

    has_domain_match = False
    matches = 0
    technical_hits = 0
    
    # Primary keywords: strict match required for domain gate
    for kw in primary_kw:
        kw_clean = kw.lower().strip()
        if not kw_clean: continue
        words = kw_clean.split()
        if len(words) > 1:
            pattern = r"\b" + r"\b(?:\W+\w+){0,4}?\W+\b".join([re.escape(w) for w in words]) + r"\b"
            if re.search(pattern, text):
                has_domain_match = True
                matches += 2
                technical_hits += 1
        else:
            if re.search(rf'\b{re.escape(kw_clean)}\b', text):
                has_domain_match = True
                matches += 1
                if len(kw_clean) > 5:
                    technical_hits += 1
                    
    # Secondary keywords: soft match, adds to score but not required
    for kw in secondary_kw:
        kw_clean = kw.lower().strip()
        if not kw_clean: continue
        if re.search(rf'\b{re.escape(kw_clean)}\b', text):
            matches += 1  # weaker contribution
                    
    # Cap contribution so long abstracts don't auto-max the score
    keyword_match = min(1.0, matches / 3.0)
    technical_match = min(1.0, technical_hits / 2.0)
    domain_match = 1.0 if has_domain_match else 0.0
    
    if not has_domain_match:
        return 0.0
        
    numeric_match = 0.0
    if domain_units:
        unit_pattern = "|".join([re.escape(u.lower()) for u in domain_units])
        if unit_pattern and re.search(rf'\b\d+(?:\.\d+)?\s*(?:{unit_pattern})\b', text):
             numeric_match = 1.0

    score = (0.25 * query_match) + (0.25 * keyword_match) + (0.25 * technical_match) + (0.15 * numeric_match) + (0.10 * domain_match)
    return min(1.0, score)

# ---------------------------------------------------------
# DOWNLOADER LOGIC
# ---------------------------------------------------------
def is_valid_pdf_content(file_path):
    """Verifies that the downloaded PDF is not just a blank error page or entirely un-ocred scan."""
    try:
         with fitz.open(file_path) as doc:
             if len(doc) == 0: return False
             
             # Fallback logic: If first 2 pages have zero text, it's likely a bad scan or paywall landing page
             text_found = False
             for i in range(min(2, len(doc))):
                  page_text = doc[i].get_text().strip()
                  if len(page_text) > 50:
                       text_found = True
                       break
             
             # Accept scans (no text) if document has more than 5 pages (actual paper, not an error slip)
             if not text_found and len(doc) < 4:
                  return False
                  
             return True
    except Exception:
         return False

def download_pdf(pdf_urls, save_path, doi=None):
    """
    Attempts to download a PDF traversing a priority-sorted list of candidate URLs.
    Falls back to next URL on HTTP errors, timeouts, or invalid PDF content.
    """
    if not isinstance(pdf_urls, list):
         pdf_urls = [pdf_urls] if pdf_urls else []
         
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    # Intercept with Unpaywall API to bypass Publisher/NIH bot-protection if DOI is available
    if doi:
        try:
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=research@academic.org"
            r = requests.get(unpaywall_url, headers=headers, timeout=10)
            if r.status_code == 200:
                best_loc = r.json().get("best_oa_location") or {}
                direct_pdf = best_loc.get("url_for_pdf")
                if direct_pdf and direct_pdf not in pdf_urls:
                    log(f"      [Unpaywall] Found direct PDF link for DOI {doi}")
                    pdf_urls.insert(0, direct_pdf)
        except Exception as e:
            pass

         
    # Optional sorting logic based on string domains.
    # We want direct URLs > PMC/NIH > Arxiv > unknown publishers
    def priority_score(url):
         url_lower = url.lower()
         if ".pdf" in url_lower and "pmc" not in url_lower: return 1
         if "pmc" in url_lower or "nih.gov" in url_lower: return 2
         if "arxiv" in url_lower: return 3
         return 4
         
    pdf_urls = sorted([url for url in set(pdf_urls) if url and "paywall" not in url.lower()], key=priority_score)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://scholar.google.com/"
    }
    
    for url in pdf_urls:
        # Adaptive timeout: Give PMC/NIH more time as they are often throttled/slow
        current_timeout = 35 if ("pmc" in url.lower() or "nih.gov" in url.lower()) else 20
        
        log(f"      Trying download URL: {url[:60]}...")
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=current_timeout, allow_redirects=True)
            if r.status_code != 200: 
                log(f"      [WARN] HTTP {r.status_code} for URL. Trying next...")
                continue
            content_type = r.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type: 
                log(f"      [WARN] URL returned HTML instead of PDF. Trying next...")
                continue
                
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            if os.path.exists(save_path) and os.path.getsize(save_path) > 10000:
                with open(save_path, 'rb') as f:
                    if f.read(4) != b'%PDF':
                        os.remove(save_path)
                        continue
                
                # Dynamic PyMuPDF semantic validation     
                if is_valid_pdf_content(save_path):
                     return True
                else:
                     log(f"      [WARN] PDF failed semantic text test (No OCR/Blank). Trying next mirror...")
                     os.remove(save_path)
                     continue
            else: # Too small, probably an error page
                if os.path.exists(save_path): os.remove(save_path)
                continue
                
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            log(f"      [WARN] Timeout or connection error on mirror. Trying next...")
            if os.path.exists(save_path): os.remove(save_path)
            continue
            
    return False

def save_paper(paper, existing_dois, existing_titles, downloaded_papers, primary_kw, secondary_kw, original_query=None, domain_units=None):
    title = paper.get('title', '')
    if not title: return False
    
    # Global Dedup (Title-Level)
    normalized_title = re.sub(r'\W+', '', title.lower())
    if normalized_title in existing_titles:
        log(f"  -> Skipping: '{title[:40]}' - Title already exists in database.")
        return False
        
    doi = paper.get('DOI', '')
    if doi and doi in existing_dois: 
        log(f"  -> Skipping: '{title[:40]}' - DOI already exists.")
        return False
        
    # V2 Adaptive Scoring Gate
    rel_score = calculate_relevance(title, paper.get('abstract', ''), primary_kw, secondary_kw, original_query, domain_units)
    if rel_score < 0.35: 
        log(f"  -> Skipping: '{title[:40]}' - Relevance Score too low ({rel_score:.2f}).")
        return False
        
    pdf_urls_candidates = paper.get('all_links', [])
    if not pdf_urls_candidates:
         # Fallback single link
         base_url = paper.get('openAccessPdf', '')
         if base_url: pdf_urls_candidates.append(base_url)
         else: return False
        
    safe_title = re.sub(r"[^\w\-]", "_", title[:50])
    hash_suffix = hashlib.md5(title.encode('utf-8')).hexdigest()[:6]
    safe_name = f"{safe_title}_{hash_suffix}.pdf"
    save_path = os.path.join(PDF_DIR, safe_name)
    
    log(f"[{paper['Source Engine']}] Found candidate: {title[:50]}... (Score: {rel_score:.2f})")
    
    if download_pdf(pdf_urls_candidates, save_path, doi=doi):
        log(f"  -> SUCCESS! Saved as {safe_name}")
        if doi: existing_dois.add(doi)
        existing_titles.add(normalized_title)
        
        paper_save = paper.copy()
        paper_save.pop('abstract', None)
        paper_save.pop('all_links', None)
        paper_save['relevance_score'] = round(rel_score, 2)
        downloaded_papers.append(paper_save)
        
        df_temp = pd.DataFrame([paper_save])
        if not os.path.exists(METADATA_FILE):
            df_temp.to_csv(METADATA_FILE, index=False)
        else:
            df_temp.to_csv(METADATA_FILE, mode='a', header=False, index=False)
        return True
        
    log(f"  -> Failed to download from any candidate mirror.")
    return False

# ---------------------------------------------------------
# API SEARCH ENGINES - IMPLEMENTATIONS
# ---------------------------------------------------------

def run_arxiv(query, limit=500):
    log("\n--- Starting arXiv API Pass ---")
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={limit}"
    
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'xml')
        results = []
        for entry in soup.find_all('entry'):
            pdf_link = ""
            for link in entry.find_all('link'):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')
                    break
            
            # ArXiv abstracts are explicitly provided
            results.append({
                "title": entry.title.text.strip() if entry.title else "",
                "authors": ", ".join(author.find('name').text for author in entry.find_all('author')),
                "year": entry.published.text[:4] if entry.published else "",
                "journal": "arXiv",
                "abstract": entry.summary.text.strip() if entry.summary else "",
                "DOI": "", # Arxiv doesn't natively supply DOI in atom feed reliably
                "openAccessPdf": pdf_link,
                "Source Engine": "arXiv"
            })
        return results
    except Exception as e:
        log(f"arXiv API error: {e}", "warning")
        return []

def run_semantic_scholar(query, limit=500):
    log("\n--- Starting Semantic Scholar API Pass ---")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    all_results = []
    headers = {"User-Agent": "BiomassResearchBot/1.0 (academic)"}
    
    try:
        for offset in range(0, limit, 100):
            params = {
                "query": query, "limit": 100, "offset": offset,
                "fields": "title,year,authors,externalIds,openAccessPdf,journal,abstract"
            }
            response = requests.get(url, params=params, headers=headers, timeout=20)
            if response.status_code == 429: # Rate limited
                log("Semantic Scholar API Rate Limit Hit... Retrying after 5s...", "warning")
                time.sleep(5)
                response = requests.get(url, params=params, headers=headers, timeout=20)
                if response.status_code == 429:
                    log("Semantic Scholar still rate limited. Skipping current source.", "warning")
                    break
                
            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            if not items: break
            
            for item in items:
                pdf_url = (item.get("openAccessPdf") or {}).get("url", "")
                doi = (item.get("externalIds") or {}).get("DOI", "")
                journal = (item.get("journal") or {}).get("name", "")
                authors = ", ".join(a["name"] for a in (item.get("authors") or []))
                
                links = [pdf_url] if pdf_url else []
                # Removed the fake CrossRef DOI .pdf endpoint that fails
                
                all_results.append({
                    "title": item.get("title", ""),
                    "authors": authors,
                    "year": item.get("year", ""),
                    "journal": journal,
                    "abstract": item.get("abstract", ""),
                    "DOI": doi,
                    "all_links": links,
                    "Source Engine": "Semantic Scholar"
                })
            time.sleep(1.5)
    except Exception as e:
        log(f"Semantic Scholar search error: {e}", "warning")
    return all_results

def run_europepmc(query, limit=500):
    log("\n--- Starting PubMed Central (Europe PMC) API Pass ---")
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": query, "format": "json", "pageSize": limit, "cursorMark": "*", "resultType": "core"}
    headers = {"User-Agent": "Mozilla/5.0 (Academic Research Pipeline)"}
    
    all_results = []
    try:
        while len(all_results) < limit:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            items = data.get("resultList", {}).get("result", [])
            for item in items:
                if item.get("isOpenAccess", "N") != "Y": continue
                pmcid = item.get("pmcid", "")
                fullTextId = item.get("id", "")
                
                # Multiple link candidate strategy
                links = []
                if pmcid:
                    links.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/")
                
                # Check for URLs in metadata links
                for l in item.get('fullTextUrlList', {}).get('fullTextUrl', []):
                    if l.get('documentStyle') == 'pdf':
                        links.append(l.get('url'))
                
                if fullTextId:
                    links.append(f"https://europepmc.org/articles/{fullTextId}?pdf=render")
                
                # We provide a list of URLs and let download_pdf try them
                best_pdf_url = links[0] if links else ""
                
                all_results.append({
                    "title": item.get("title", ""),
                    "authors": item.get("authorString", ""),
                    "year": item.get("pubYear", ""),
                    "journal": item.get("journalTitle", ""),
                    "abstract": item.get("abstractText", ""),
                    "DOI": item.get("doi", ""),
                    "openAccessPdf": best_pdf_url,
                    "all_links": links, # Store for potential retry
                    "Source Engine": "PubMed Central / Europe PMC"
                })
                
            next_cursor = data.get("nextCursorMark", "")
            if not items or params["cursorMark"] == next_cursor: break
            params["cursorMark"] = next_cursor
            time.sleep(1)
            
    except Exception as e:
        log(f"Europe PMC search error: {e}", "warning")
    return all_results

def run_crossref(query, limit=500):
    log("\n--- Starting CrossRef API Pass ---")
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": limit, "select": "DOI,title,author,link,abstract,published"}
    all_results = []
    
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        for item in items:
            pdf_url = ""
            for link in item.get('link', []):
                if link.get('content-type') == 'application/pdf':
                    pdf_url = link.get('URL', '')
                    break
            
            title = item.get("title", [""])[0]
            authors = ", ".join(f"{a.get('given','')} {a.get('family','')}" for a in item.get("author", []))
            
            pub = item.get("published", {}).get("date-parts", [[""]])
            year = pub[0][0] if pub and pub[0] else ""
            
            all_results.append({
                "title": title,
                "authors": authors,
                "year": year,
                "journal": "CrossRef",
                "abstract": item.get("abstract", ""),
                "DOI": item.get("DOI", ""),
                "all_links": [pdf_url] if pdf_url else [],
                "Source Engine": "CrossRef"
            })
    except Exception as e:
        log(f"CrossRef API error: {e}", "warning")
    return all_results

def run_core(query, limit=500):
    log("\n--- Starting CORE API Pass ---")
    url = "https://api.core.ac.uk/v3/search/works"
    params = {"q": query, "limit": 100} # Max CORE chunk limit
    all_results = []
    
    try:
        for offset in range(0, limit, 100):
            params["offset"] = offset
            r = requests.get(url, params=params, timeout=40)
            if r.status_code == 401 or r.status_code == 429:
                log("CORE API key required or Rate Limited. Skipping CORE.", "warning")
                break
            r.raise_for_status()
            
            data = r.json().get("results", [])
            if not data: break
            
            for item in data:
                pdf_url = item.get("downloadUrl")
                if not pdf_url: continue
                doi = next((i["identifier"] for i in (item.get("identifiers") or []) if i.get("identifier", "").startswith("10.")), "")
                
                all_results.append({
                    "title": item.get("title", ""),
                    "authors": ", ".join(a.get("name", "") for a in item.get("authors", [])),
                    "year": item.get("yearPublished", ""),
                    "journal": item.get("publisher", ""),
                    "abstract": item.get("abstract", ""),
                    "DOI": doi,
                    "all_links": [pdf_url],
                    "Source Engine": "CORE"
                })
            time.sleep(1)
    except requests.exceptions.ReadTimeout:
        log("CORE API request timed out. Continuing to next engine.", "warning")
    except Exception as e:
        log(f"CORE API error: {e}", "warning")
    return all_results


def run_google_scholar_scraper(query, limit=500):
    log("\n--- Starting Google Scholar HTML Scraper Pass ---")
    # Scholar severely limits without proxies. This implements the 10-page run.
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
    all_results = []
    
    for page in range(0, limit, 10):
        url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}&start={page}"
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 429:
                log("Google Scholar Blocked IP (HTTP 429). Terminating Scholar scrape.", "warning")
                break
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            
            blocks = soup.select('.gs_ri')
            if not blocks: break
                
            for item in blocks:
                title_elem = item.select_one('.gs_rt a') or item.select_one('.gs_rt')
                title = title_elem.text.replace("[HTML]", "").replace("[PDF]", "").strip() if title_elem else ""
                
                pdf_link = ""
                pdf_div = item.parent.select_one('.gs_or_ggsm a')
                if pdf_div and pdf_div.get('href', '').endswith('.pdf'):
                    pdf_link = pdf_div.get('href')
                
                # We skip missing PDFs aggressively on Scholar to avoid getting blocked tracking HTML pages
                if not pdf_link: continue 
                    
                abstract = getattr(item.select_one('.gs_rs'), 'text', '')
                
                all_results.append({
                    "title": title,
                    "authors": "", "year": "", "journal": "Google Scholar",
                    "abstract": abstract,
                    "DOI": "", 
                    "all_links": [pdf_link],
                    "Source Engine": "Google Scholar"  
                })
            log("Sleeping 3 seconds before next Scholar pagination click...")
            time.sleep(3)
        except Exception as e:
            log(f"Google Scholar error: {e}", "warning")
            break
            
    return all_results

# ---------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------
def main(query, keywords_dict, target_count=200):
    """
    Main orchestrator. keywords_dict must follow: {"primary": [...], "secondary": [...]}
    """
    primary_kw = keywords_dict.get("primary", [])
    secondary_kw = keywords_dict.get("secondary", [])
    all_kw = primary_kw + secondary_kw
    
    log("\n" + "="*50)
    log("   Federated Academic API Paper Downloader        ")
    log("==================================================")
    log(f"Query: {query}")
    log(f"Target Download Count: {target_count}")
    log(f"Primary Keywords [{len(primary_kw)}]: {', '.join(primary_kw)}")
    log(f"Secondary Keywords [{len(secondary_kw)}]: {', '.join(secondary_kw)}")
    
    # We search a larger pool to ensure we hit the target OA download count
    search_buffer = max(target_count * 5, 100)
    
    existing_dois = set()
    existing_titles = set()
    if os.path.exists(METADATA_FILE):
        try:
            df_existing = pd.read_csv(METADATA_FILE)
            if 'DOI' in df_existing.columns:
                existing_dois = set(df_existing['DOI'].dropna().astype(str))
            if 'title' in df_existing.columns:
                # Add normalized titles
                existing_titles = set(df_existing['title'].dropna().astype(str).apply(lambda x: re.sub(r'\W+', '', x.lower())))
            log(f"Loaded {len(existing_dois)} DOIs and {len(existing_titles)} Titles from memory.")
        except Exception:
            pass
            
    downloaded_papers = []
    
    # Load domain units dynamically for relevance scoring
    domain_units = set()
    schema_sys = load_domain_schema()
    if schema_sys and hasattr(schema_sys, 'schema'):
         for v in schema_sys.schema.values():
              domain_units.update([u.lower() for u in v.get("units", [])])
    
    # Run the federated search in Optimised Priority Order
    try:
        engines = [
            ("CORE", run_core),
            ("arXiv", run_arxiv),
            ("CrossRef", run_crossref),
            ("Semantic Scholar", run_semantic_scholar),
            ("Europe PMC", run_europepmc),
            ("Google Scholar", run_google_scholar_scraper) # Fallback to avoid aggressive rate limiting
        ]
        
        for engine_name, engine_func in engines:
            if len(downloaded_papers) >= target_count:
                break
                
            log(f"\n--- Querying {engine_name} (Buffer: {search_buffer}) ---")
            for paper in engine_func(query, search_buffer):
                # GLOBAL RATE LIMIT CONTROL
                time.sleep(1.0) # Prevent bursting 
                
                if save_paper(paper, existing_dois, existing_titles, downloaded_papers, primary_kw, secondary_kw, query, domain_units):
                    if len(downloaded_papers) >= target_count:
                        log(f"\n[INFO] Target count of {target_count} reached. Stopping.")
                        break
            
    except KeyboardInterrupt:
        log("\nFederated extraction manually terminated by User.")
        
    log("==================================================")
    log(f"Job Complete. Successfully downloaded {len(downloaded_papers)} new OA papers.")
    log(f"Metadata recorded.")
    log("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Academic API Downloader")
    parser.add_argument("--query", type=str, required=True, help="Search query (e.g. 'Mxene Supercapacitors')")
    parser.add_argument("--keywords", type=str, required=True, 
        help="Structured JSON keywords from keyword_extractor: '{\"primary\": [...], \"secondary\": [...]}'. Fallback: comma-separated string.")
    parser.add_argument("--limit", type=int, default=200, help="Max items per API")
    parser.add_argument("--workspace", type=str, help="Workspace directory for this session")
    
    args = parser.parse_args()
    
    # Initialize workspace paths before main
    setup_paths(args.workspace)
    
    query = args.query
    
    # Support both structured JSON dict and legacy comma-separated keyword strings
    try:
        keywords_dict = json.loads(args.keywords)
        if not isinstance(keywords_dict, dict):
            raise ValueError("Not a dict")
    except (json.JSONDecodeError, ValueError):
        # Legacy fallback: treat as comma separated list
        raw = [k.strip() for k in args.keywords.split(",") if k.strip()]
        mid = len(raw) // 2
        keywords_dict = {"primary": raw[:mid] or raw, "secondary": raw[mid:]}
        
    main(query, keywords_dict, args.limit)
