#!/usr/bin/env python
"""
Simple Web Search and Content Extraction Script

This script:
1. Performs a search on DuckDuckGo
2. Takes the top N results without any filtering
3. Extracts the full content from each URL
4. Displays each URL and its content in paragraph format
   (skipping any results where content could not be fetched)
"""

import sys
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import time
import urllib3
import argparse

# Disable SSL warnings to avoid cluttering the output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def search_duckduckgo(query, num_results=10):
    """
    Search DuckDuckGo and return top results exactly as given
    """
    print(f"Searching DuckDuckGo for: {query}")
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=num_results))
        print(f"Found {len(results)} results")
        return results
    except Exception as e:
        print(f"Error searching DuckDuckGo: {e}")
        return []


def extract_content(url):
    """
    Extract full content from a URL with fallback for SSL errors
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        if "SSL" in str(e):
            try:
                response = requests.get(url, headers=headers, timeout=15, verify=False)
                response.raise_for_status()
            except Exception:
                return None
        else:
            return None

    content_type = response.headers.get('Content-Type', '').lower()
    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
        return None

    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        for elem in soup(['script', 'style']):
            elem.extract()
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Simple web search and content extraction')
    parser.add_argument('query', nargs='*', help='Search query')
    parser.add_argument('--results', '-r', type=int, default=10,
                        help='Number of search results to process')
    args = parser.parse_args()

    query = ' '.join(args.query) if args.query else input("Enter your search query: ")
    if not query:
        print("Error: Search query is required")
        sys.exit(1)

    results = search_duckduckgo(query, args.results)
    if not results:
        print("No results found. Please try another query.")
        sys.exit(1)

    # Extract content
    for i, res in enumerate(results, start=1):
        url = res.get('href')
        if not url:
            res['content'] = None
            continue
        time.sleep(1)  # polite delay
        res['content'] = extract_content(url)

    # Print out URLs and their extracted content
    print(f"\n\nContent for '{query}':\n")
    for res in results:
        url = res.get('href')
        content = res.get('content')
        if not content:
            continue
        print(f"URL: {url}\n")
        print(f"Content:\n{content}\n")
        print("-"*80)

    sys.exit(0)

if __name__ == "__main__":
    # Ensure dependencies
    missing = []
    for pkg in ["bs4", "duckduckgo_search", "requests", "urllib3"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing packages: pip install %s" % ' '.join(missing))
        sys.exit(1)

    main()
