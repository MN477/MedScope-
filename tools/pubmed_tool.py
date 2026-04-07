"""PubMed Article Fetcher - Multi-step pipeline for retrieving scientific articles."""

import requests
import xml.etree.ElementTree as ET
import logging
import time
import re
import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Article:
    """Unified article data structure."""
    pmid: str
    pmcid: Optional[str]
    title: str
    content_type: str  # "full_text" or "abstract"
    content: str

    def to_dict(self) -> Dict:
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "title": self.title,
            "content_type": self.content_type,
            "content": self.content,
        }


# ============================================================================
# HTTP Client
# ============================================================================


class HTTPClient:
    """Simple HTTP client with rate limiting."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self):
        self.api_key = os.getenv("NCBI_API_KEY")
        self.last_request = 0
        self.delay = 0.34 if self.api_key else 0.34  # ~3 req/s

    def get(self, endpoint: str, params: Dict) -> str:
        """Make rate-limited GET request."""
        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        
        # Add API key if available
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        self.last_request = time.time()
        return response.text


# ============================================================================
# Component 1: ArticleSearcher
# ============================================================================


class ArticleSearcher:
    """Searches PubMed for articles."""
    
    def __init__(self, http_client: HTTPClient):
        self.http = http_client

    def search(self, query: str, max_results: int = 50) -> List[str]:
        """Search PubMed and return PMIDs."""
        # Add field tag for better relevance
        if "[" not in query:
            query = f"{query}[Title/Abstract]"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        response = self.http.get("esearch.fcgi", params)
        data = json.loads(response)
        pmids = data.get("esearchresult", {}).get("idlist", [])
        
        logger.info(f"Found {len(pmids)} articles for '{query}'")
        return pmids


# ============================================================================
# Component 2: PMCAvailabilityChecker
# ============================================================================


class PMCAvailabilityChecker:
    """Checks which articles have PMC full-text."""
    
    def __init__(self, http_client: HTTPClient):
        self.http = http_client
        self.cache = {}

    def check(self, pmids: List[str]) -> Dict[str, Optional[str]]:
        """Return PMID -> PMCID mapping."""
        if not pmids:
            return {}
        
        # Check cache
        uncached = [p for p in pmids if p not in self.cache]
        
        if uncached:
            params = {
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": ",".join(uncached),
                "retmode": "xml"
            }
            
            response = self.http.get("elink.fcgi", params)
            root = ET.fromstring(response)
            
            # Parse mappings
            for linkset in root.findall(".//LinkSet"):
                pmid_elem = linkset.find("IdList/Id")
                pmid = pmid_elem.text if pmid_elem is not None else None
                
                pmcid = None
                for linksetdb in linkset.findall("LinkSetDb"):
                    dbto = linksetdb.find("DbTo")
                    if dbto is not None and dbto.text == "pmc":
                        link_id = linksetdb.find("Link/Id")
                        if link_id is not None:
                            pmcid = f"PMC{link_id.text}"
                            break
                
                if pmid:
                    self.cache[pmid] = pmcid
            
            # Mark uncached PMIDs without mapping as None
            for pmid in uncached:
                if pmid not in self.cache:
                    self.cache[pmid] = None
        
        return {pmid: self.cache.get(pmid) for pmid in pmids}


# ============================================================================
# Component 3: FullTextFetcher
# ============================================================================


class FullTextFetcher:
    """Fetches full-text from PMC."""
    
    def __init__(self, http_client: HTTPClient):
        self.http = http_client

    def fetch(self, pmcid: str) -> Dict[str, str]:
        """Fetch full-text article."""
        params = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "full",
            "retmode": "xml"
        }
        
        response = self.http.get("efetch.fcgi", params)
        root = ET.fromstring(response)
        
        # Extract title
        title_elem = root.find(".//article-title")
        title = self._get_text(title_elem)
        
        # Extract abstract
        abstract_elem = root.find(".//abstract")
        abstract = self._get_text(abstract_elem)
        
        # Extract body sections
        sections = []
        for section_name in ["Introduction", "Results", "Discussion", "Conclusion"]:
            text = self._find_section(root, section_name)
            if text:
                sections.append(f"{section_name}: {text}")
        
        content = " ".join(sections) if sections else abstract
        
        return {"title": title, "content": content}

    def _get_text(self, elem) -> str:
        """Extract all text from element."""
        if elem is None:
            return ""
        return " ".join(elem.itertext()).strip()

    def _find_section(self, root, section_name: str) -> Optional[str]:
        """Find section by title."""
        for sec in root.findall(".//sec"):
            title_elem = sec.find("title")
            if title_elem is not None and section_name.lower() in title_elem.text.lower():
                return self._get_text(sec)
        return None


# ============================================================================
# Component 4: AbstractFetcher
# ============================================================================


class AbstractFetcher:
    """Fetches abstracts from PubMed."""
    
    def __init__(self, http_client: HTTPClient):
        self.http = http_client

    def fetch(self, pmid: str) -> Dict[str, str]:
        """Fetch abstract."""
        params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml"
        }
        
        response = self.http.get("efetch.fcgi", params)
        root = ET.fromstring(response)
        
        # Extract title
        title_elem = root.find(".//ArticleTitle")
        title = self._get_text(title_elem)
        
        # Extract abstract with labels
        abstract_parts = []
        for abstract_text in root.findall(".//AbstractText"):
            label = abstract_text.get("Label", "")
            text = self._get_text(abstract_text)
            
            if text:
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        
        content = " ".join(abstract_parts)
        
        return {"title": title, "content": content}

    def _get_text(self, elem) -> str:
        """Extract all text from element."""
        if elem is None:
            return ""
        return " ".join(elem.itertext()).strip()


# ============================================================================
# Component 5: PubMedPipeline
# ============================================================================


class PubMedPipeline:
    """Orchestrates the complete pipeline."""
    
    def __init__(self):
        self.http = HTTPClient()
        self.searcher = ArticleSearcher(self.http)
        self.pmc_checker = PMCAvailabilityChecker(self.http)
        self.full_text_fetcher = FullTextFetcher(self.http)
        self.abstract_fetcher = AbstractFetcher(self.http)

    def fetch_articles(self, query: str, max_results: int = 100) -> List[Dict]:
        """Execute pipeline: search → check PMC → fetch content."""
        articles = []
        
        # Step 1: Search
        pmids = self.searcher.search(query, max_results)
        if not pmids:
            return []
        
        # Step 2: Check PMC availability
        pmc_mappings = self.pmc_checker.check(pmids)
        
        # Step 3-4: Fetch content
        for pmid in pmids:
            pmcid = pmc_mappings.get(pmid)
            
            try:
                if pmcid:
                    # Try full-text
                    try:
                        data = self.full_text_fetcher.fetch(pmcid)
                        article = Article(
                            pmid=pmid,
                            pmcid=pmcid,
                            title=data["title"],
                            content_type="full_text",
                            content=data["content"]
                        )
                    except:
                        # Fallback to abstract
                        data = self.abstract_fetcher.fetch(pmid)
                        article = Article(
                            pmid=pmid,
                            pmcid=pmcid,
                            title=data["title"],
                            content_type="abstract",
                            content=data["content"]
                        )
                else:
                    # Fetch abstract only
                    data = self.abstract_fetcher.fetch(pmid)
                    article = Article(
                        pmid=pmid,
                        pmcid=None,
                        title=data["title"],
                        content_type="abstract",
                        content=data["content"]
                    )
                
                articles.append(article.to_dict())
                
            except Exception as e:
                logger.error(f"Failed to fetch {pmid}: {e}")
                continue
        
        # Log summary
        full_text = sum(1 for a in articles if a["content_type"] == "full_text")
        abstract = sum(1 for a in articles if a["content_type"] == "abstract")
        logger.info(f"Retrieved {len(articles)} articles ({full_text} full-text, {abstract} abstract)")
        
        return articles


# ============================================================================
# Main
# ============================================================================


def main():
    """Example usage."""
    pipeline = PubMedPipeline()
    articles = pipeline.fetch_articles("diabetes", max_results=5)
    
    print(f"\nRetrieved {len(articles)} articles\n")
    
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['title'][:80]}...")
        print(f"   PMID: {article['pmid']} | PMCID: {article['pmcid']} | Type: {article['content_type']}")
        print(f"   Content: {article['content'][:150]}...\n")


if __name__ == "__main__":
    main()
