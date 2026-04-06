"""PubMed Article Fetcher - Multi-step pipeline for retrieving scientific articles."""

import requests
import xml.etree.ElementTree as ET
import logging
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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

    def __post_init__(self):
        """Validate article data."""
        if not self.pmid or not re.match(r"^\d+$", self.pmid):
            raise ValueError(f"Invalid PMID format: {self.pmid}")

        if self.pmcid is not None and not re.match(r"^PMC\d+$", self.pmcid):
            raise ValueError(f"Invalid PMCID format: {self.pmcid}")

        if not self.title or not isinstance(self.title, str):
            raise ValueError("Title must be a non-empty string")

        if self.content_type not in ("full_text", "abstract"):
            raise ValueError(
                f"content_type must be 'full_text' or 'abstract', got {self.content_type}"
            )

        if not self.content or not isinstance(self.content, str):
            raise ValueError("Content must be a non-empty string")

    def to_dict(self) -> Dict:
        """Convert article to dictionary."""
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "title": self.title,
            "content_type": self.content_type,
            "content": self.content,
        }


@dataclass
class SearchResult:
    """Search result metadata."""

    pmids: List[str]
    total_count: int
    query: str


@dataclass
class PMCMapping:
    """PMID to PMCID mapping."""

    pmid_to_pmcid: Dict[str, Optional[str]]


@dataclass
class ArticleContent:
    """Article content with sections."""

    title: str
    abstract: str
    introduction: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None


# ============================================================================
# HTTP Client with Retry Logic
# ============================================================================


class HTTPClient:
    """HTTP client with timeout, retry, and rate limiting."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    TIMEOUT = 30
    MAX_RETRIES = 3
    RATE_LIMIT = 3  # requests per second
    USER_AGENT = "PubMedArticleFetcher/1.0 (Python)"

    def __init__(self):
        """Initialize HTTP client."""
        self.last_request_time = 0
        self.request_queue = []

    def _apply_rate_limit(self):
        """Apply rate limiting (3 requests/second)."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.RATE_LIMIT

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_delay = 2 ** attempt
        jitter = base_delay * 0.1 * (time.time() % 1)
        return base_delay + jitter

    def get(self, endpoint: str, params: Dict) -> str:
        """Make GET request with retry logic."""
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"User-Agent": self.USER_AGENT}

        for attempt in range(self.MAX_RETRIES):
            try:
                self._apply_rate_limit()
                logger.debug(f"GET {endpoint} with params {params}")

                response = requests.get(
                    url, params=params, headers=headers, timeout=self.TIMEOUT
                )

                if response.status_code == 429 or response.status_code == 503:
                    if attempt < self.MAX_RETRIES - 1:
                        backoff = self._exponential_backoff(attempt)
                        logger.warning(
                            f"Rate limited (status {response.status_code}), retrying in {backoff:.2f}s"
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        raise Exception(
                            f"API rate limit exceeded after {self.MAX_RETRIES} retries"
                        )

                response.raise_for_status()
                logger.debug(f"Response status: {response.status_code}")
                return response.text

            except requests.Timeout:
                if attempt < self.MAX_RETRIES - 1:
                    backoff = self._exponential_backoff(attempt)
                    logger.warning(f"Timeout, retrying in {backoff:.2f}s")
                    time.sleep(backoff)
                    continue
                else:
                    raise Exception(f"Request timeout after {self.MAX_RETRIES} retries")

            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                raise

        raise Exception("Max retries exceeded")


# ============================================================================
# Component 1: ArticleSearcher
# ============================================================================


class ArticleSearcher:
    """Searches PubMed for articles matching a query."""

    def __init__(self, http_client: HTTPClient):
        """Initialize searcher."""
        self.http_client = http_client

    def search(self, query: str, max_results: int = 100) -> SearchResult:
        """Search PubMed for articles."""
        if not query or not isinstance(query, str) or query.strip() == "":
            raise ValueError("Search query must be a non-empty string")

        if not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("max_results must be a positive integer")

        # Sanitize query
        query = self._sanitize_query(query)

        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}

        try:
            response_text = self.http_client.get("esearch.fcgi", params)
            data = self._parse_esearch_response(response_text)

            pmids = data.get("pmids", [])
            total_count = int(data.get("total_count", 0))

            logger.info(
                f"Search for '{query}' returned {len(pmids)} PMIDs (total: {total_count})"
            )

            return SearchResult(pmids=pmids, total_count=total_count, query=query)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _sanitize_query(self, query: str) -> str:
        """Sanitize search query to prevent injection."""
        # Remove potentially dangerous characters but preserve search intent
        query = re.sub(r'[<>"]', "", query)
        return query.strip()

    def _parse_esearch_response(self, response_text: str) -> Dict:
        """Parse esearch JSON response."""
        try:
            import json
            data = json.loads(response_text)
            
            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])
            total_count = int(result.get("count", 0))

            return {"pmids": pmids, "total_count": total_count}

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse esearch response: {e}")
            raise


# ============================================================================
# Component 2: PMCAvailabilityChecker
# ============================================================================


class PMCAvailabilityChecker:
    """Checks which articles have PMC availability."""

    def __init__(self, http_client: HTTPClient):
        """Initialize checker."""
        self.http_client = http_client
        self.cache = {}

    def check_pmc_availability(self, pmids: List[str]) -> Dict[str, Optional[str]]:
        """Check PMC availability for PMIDs."""
        if not pmids:
            return {}

        # Validate PMIDs
        for pmid in pmids:
            if not re.match(r"^\d+$", pmid):
                raise ValueError(f"Invalid PMID format: {pmid}")

        # Check cache first
        uncached_pmids = [pmid for pmid in pmids if pmid not in self.cache]

        if uncached_pmids:
            # Batch request (NCBI allows up to 200 IDs per request)
            batch_size = 200
            for i in range(0, len(uncached_pmids), batch_size):
                batch = uncached_pmids[i : i + batch_size]
                self._fetch_pmc_mappings(batch)

        # Return all mappings (cached + newly fetched)
        result = {}
        for pmid in pmids:
            result[pmid] = self.cache.get(pmid)

        return result

    def _fetch_pmc_mappings(self, pmids: List[str]):
        """Fetch PMC mappings for a batch of PMIDs."""
        pmid_str = ",".join(pmids)
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid_str,
            "retmode": "json",
        }

        try:
            response_text = self.http_client.get("elink.fcgi", params)
            mappings = self._parse_elink_response(response_text)

            # Update cache
            for pmid in pmids:
                self.cache[pmid] = mappings.get(pmid)

            logger.debug(f"Fetched PMC mappings for {len(pmids)} PMIDs")

        except Exception as e:
            logger.error(f"Failed to fetch PMC mappings: {e}")
            # Mark as not available on error
            for pmid in pmids:
                self.cache[pmid] = None

    def _parse_elink_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """Parse elink XML response."""
        mappings = {}

        try:
            root = ET.fromstring(response_text)

            # Parse LinkSet elements
            for linkset in root.findall(".//LinkSet"):
                pmid_elem = linkset.find(".//IdList/Id")
                pmid = pmid_elem.text if pmid_elem is not None else None

                # Find PMC ID in LinkSetDb
                pmcid = None
                for linksetdb in linkset.findall(".//LinkSetDb"):
                    link_name = linksetdb.find("LinkName")
                    if link_name is not None and "pmc" in link_name.text.lower():
                        id_list = linksetdb.find("Link/Id")
                        if id_list is not None:
                            pmcid = f"PMC{id_list.text}"
                            break

                if pmid:
                    mappings[pmid] = pmcid

            return mappings

        except ET.ParseError as e:
            logger.error(f"Failed to parse elink response: {e}")
            return {}


# ============================================================================
# Component 3: FullTextFetcher
# ============================================================================


class FullTextFetcher:
    """Fetches full-text articles from PMC."""

    def __init__(self, http_client: HTTPClient):
        """Initialize fetcher."""
        self.http_client = http_client

    def fetch_full_text(self, pmcid: str) -> ArticleContent:
        """Fetch full-text article from PMC."""
        if not re.match(r"^PMC\d+$", pmcid):
            raise ValueError(f"Invalid PMCID format: {pmcid}")

        params = {"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"}

        try:
            response_text = self.http_client.get("efetch.fcgi", params)
            content = self._parse_pmc_response(response_text)
            logger.debug(f"Fetched full-text for {pmcid}")
            return content

        except Exception as e:
            logger.error(f"Failed to fetch full-text for {pmcid}: {e}")
            raise

    def _parse_pmc_response(self, response_text: str) -> ArticleContent:
        """Parse PMC XML response."""
        try:
            root = ET.fromstring(response_text)

            # Extract title
            title_elem = root.find(".//article-title")
            title = self._extract_text(title_elem) if title_elem is not None else ""

            # Extract abstract
            abstract_elem = root.find(".//abstract")
            abstract = self._extract_text(abstract_elem) if abstract_elem is not None else ""

            # Extract body sections
            introduction = self._extract_section(root, "Introduction")
            results = self._extract_section(root, "Results")
            discussion = self._extract_section(root, "Discussion")
            conclusion = self._extract_section(root, "Conclusion")

            return ArticleContent(
                title=title,
                abstract=abstract,
                introduction=introduction,
                results=results,
                discussion=discussion,
                conclusion=conclusion,
            )

        except ET.ParseError as e:
            logger.warning(f"Failed to parse PMC XML: {e}")
            raise

    def _extract_text(self, element) -> str:
        """Extract text from XML element."""
        if element is None:
            return ""

        text_parts = []
        for text in element.itertext():
            text = text.strip()
            if text:
                text_parts.append(text)

        return " ".join(text_parts)

    def _extract_section(self, root, section_title: str) -> Optional[str]:
        """Extract specific section from body."""
        for sec in root.findall(".//body/sec"):
            title_elem = sec.find("title")
            if title_elem is not None and section_title.lower() in title_elem.text.lower():
                return self._extract_text(sec)

        return None


# ============================================================================
# Component 4: AbstractFetcher
# ============================================================================


class AbstractFetcher:
    """Fetches abstracts from PubMed."""

    def __init__(self, http_client: HTTPClient):
        """Initialize fetcher."""
        self.http_client = http_client

    def fetch_abstract(self, pmid: str) -> ArticleContent:
        """Fetch abstract from PubMed."""
        if not re.match(r"^\d+$", pmid):
            raise ValueError(f"Invalid PMID format: {pmid}")

        params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml"}

        try:
            response_text = self.http_client.get("efetch.fcgi", params)
            content = self._parse_pubmed_response(response_text)
            logger.debug(f"Fetched abstract for {pmid}")
            return content

        except Exception as e:
            logger.error(f"Failed to fetch abstract for {pmid}: {e}")
            raise

    def _parse_pubmed_response(self, response_text: str) -> ArticleContent:
        """Parse PubMed XML response."""
        try:
            root = ET.fromstring(response_text)

            # Extract title
            title_elem = root.find(".//ArticleTitle")
            title = self._extract_text(title_elem) if title_elem is not None else ""

            # Extract abstract
            abstract_parts = []
            for abstract_text in root.findall(".//AbstractText"):
                text = self._extract_text(abstract_text)
                if text:
                    abstract_parts.append(text)

            abstract = " ".join(abstract_parts)

            return ArticleContent(title=title, abstract=abstract)

        except ET.ParseError as e:
            logger.warning(f"Failed to parse PubMed XML: {e}")
            raise

    def _extract_text(self, element) -> str:
        """Extract text from XML element."""
        if element is None:
            return ""

        text_parts = []
        for text in element.itertext():
            text = text.strip()
            if text:
                text_parts.append(text)

        return " ".join(text_parts)


# ============================================================================
# Component 5: PubMedPipeline (Orchestrator)
# ============================================================================


class PubMedPipeline:
    """Orchestrates the complete article fetching pipeline."""

    def __init__(self):
        """Initialize pipeline."""
        self.http_client = HTTPClient()
        self.searcher = ArticleSearcher(self.http_client)
        self.pmc_checker = PMCAvailabilityChecker(self.http_client)
        self.full_text_fetcher = FullTextFetcher(self.http_client)
        self.abstract_fetcher = AbstractFetcher(self.http_client)

    def fetch_articles(self, query: str, max_results: int = 100) -> List[Dict]:
        """Execute complete pipeline: search → check PMC → fetch content."""
        articles = []
        errors = []

        try:
            # Step 1: Search for articles
            logger.info(f"Step 1: Searching for '{query}'")
            search_result = self.searcher.search(query, max_results)
            pmids = search_result.pmids

            if not pmids:
                logger.info("No articles found")
                return []

            logger.info(f"Found {len(pmids)} articles")

            # Step 2: Check PMC availability
            logger.info("Step 2: Checking PMC availability")
            pmc_mappings = self.pmc_checker.check_pmc_availability(pmids)

            # Step 3 & 4: Fetch content (full-text or abstract)
            logger.info("Step 3-4: Fetching article content")

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}

                for pmid in pmids:
                    pmcid = pmc_mappings.get(pmid)

                    if pmcid:
                        # Try full-text first
                        future = executor.submit(
                            self._fetch_with_fallback, pmid, pmcid
                        )
                    else:
                        # Fetch abstract directly
                        future = executor.submit(self._fetch_abstract_only, pmid)

                    futures[future] = pmid

                # Collect results
                for future in as_completed(futures):
                    pmid = futures[future]
                    try:
                        article = future.result()
                        if article:
                            articles.append(article)
                    except Exception as e:
                        error_msg = f"Failed to fetch article {pmid}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

            # Log summary
            logger.info(
                f"Pipeline complete: {len(articles)} articles retrieved, {len(errors)} errors"
            )

            return articles

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _fetch_with_fallback(self, pmid: str, pmcid: str) -> Optional[Dict]:
        """Fetch full-text with fallback to abstract."""
        try:
            # Try full-text
            content = self.full_text_fetcher.fetch_full_text(pmcid)
            article = Article(
                pmid=pmid,
                pmcid=pmcid,
                title=content.title,
                content_type="full_text",
                content=self._format_full_text(content),
            )
            return article.to_dict()

        except Exception as e:
            logger.warning(f"Full-text fetch failed for {pmid}, falling back to abstract: {e}")

            try:
                # Fallback to abstract
                content = self.abstract_fetcher.fetch_abstract(pmid)
                article = Article(
                    pmid=pmid,
                    pmcid=pmcid,
                    title=content.title,
                    content_type="abstract",
                    content=content.abstract,
                )
                return article.to_dict()

            except Exception as e2:
                logger.error(f"Abstract fetch also failed for {pmid}: {e2}")
                return None

    def _fetch_abstract_only(self, pmid: str) -> Optional[Dict]:
        """Fetch abstract only."""
        try:
            content = self.abstract_fetcher.fetch_abstract(pmid)
            article = Article(
                pmid=pmid,
                pmcid=None,
                title=content.title,
                content_type="abstract",
                content=content.abstract,
            )
            return article.to_dict()

        except Exception as e:
            logger.error(f"Failed to fetch abstract for {pmid}: {e}")
            return None

    def _format_full_text(self, content: ArticleContent) -> str:
        """Format full-text content."""
        sections = []

        if content.introduction:
            sections.append(f"Introduction: {content.introduction}")
        if content.results:
            sections.append(f"Results: {content.results}")
        if content.discussion:
            sections.append(f"Discussion: {content.discussion}")
        if content.conclusion:
            sections.append(f"Conclusion: {content.conclusion}")

        return " ".join(sections) if sections else content.abstract


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Example usage of PubMedPipeline."""
    pipeline = PubMedPipeline()

    # Example search
    query = "type 2 diabetes"
    articles = pipeline.fetch_articles(query, max_results=5)

    print(f"\nRetrieved {len(articles)} articles for '{query}':\n")
    for article in articles:
        print(f"PMID: {article['pmid']}")
        print(f"PMCID: {article['pmcid']}")
        print(f"Title: {article['title']}")
        print(f"Content Type: {article['content_type']}")
        print(f"Content: {article['content'][:200]}...\n")


if __name__ == "__main__":
    main()
