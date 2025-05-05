import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
from urllib.parse import urljoin

def scrape_shl_assessments():
    """Enhanced SHL catalog scraper with pagination support"""
    BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    assessments = []
    next_page_url = BASE_URL
    while next_page_url:
        try:
            print(f"🔄 Fetching SHL catalog page: {next_page_url}")
            response = requests.get(next_page_url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract assessment data from table rows
            table = soup.find('table')
            if not table:
                print("⚠️ No table found on the page")
                break
            rows = table.find_all('tr')
            if not rows or len(rows) <= 1:
                print("⚠️ No data rows found in the table")
                break
            for row in rows[1:]:  # Skip header row
                try:
                    cols = row.find_all('td')
                    if len(cols) < 3:
                        continue
                    link_elem = cols[0].find('a', href=True)
                    name = link_elem.text.strip() if link_elem else cols[0].text.strip()
                    url = urljoin(BASE_URL, link_elem['href']) if link_elem else ""
                    duration_text = ''.join(filter(str.isdigit, cols[1].text)) or "0"
                    duration = int(duration_text)
                    test_type_text = cols[2].text.strip()
                    test_types = [t.strip() for t in test_type_text.split(',')] if test_type_text else []
                    row_text = row.text.lower()
                    remote_support = "Yes" if any(keyword in row_text for keyword in ["remote", "online", "virtual"]) else "No"
                    adaptive_support = "Yes" if "adaptive" in row_text else "No"
                    assessment = {
                        "name": name,
                        "url": url,
                        "duration": duration,
                        "test_type": test_types,
                        "remote_support": remote_support,
                        "adaptive_support": adaptive_support
                    }
                    assessments.append(assessment)
                except Exception as e:
                    print(f"⚠️ Error parsing row: {e}")
                    continue
            # Find next page link
            next_page_link = soup.find('a', string='Next')
            if next_page_link and 'href' in next_page_link.attrs:
                next_page_url = urljoin(BASE_URL, next_page_link['href'])
            else:
                next_page_url = None
        except Exception as e:
            print(f"❌ Critical error: {e}")
            break
    # Save results
    output_path = Path('data/raw/assessments.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(assessments, f, indent=2)
    print(f"✅ Successfully scraped {len(assessments)} assessments")
    return assessments

if __name__ == "__main__":
    scrape_shl_assessments()
