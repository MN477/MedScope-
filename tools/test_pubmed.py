"""Simple test script for PubMed Article Fetcher."""

from pubmed_tool import PubMedPipeline


def test_search():
    """Test basic search functionality."""
    pipeline = PubMedPipeline()
    
    print("Testing PubMed Article Fetcher...\n")
    
    # Test search
    query = "diabetes"
    print(f"Searching for: '{query}'")
    articles = pipeline.fetch_articles(query, max_results=3)
    
    print(f"\nRetrieved {len(articles)} articles:\n")
    
    for i, article in enumerate(articles, 1):
        print(f"Article {i}:")
        print(f"  PMID: {article['pmid']}")
        print(f"  PMCID: {article['pmcid']}")
        print(f"  Title: {article['title'][:80]}...")
        print(f"  Type: {article['content_type']}")
        print(f"  Content: {article['content'][:100]}...\n")


if __name__ == "__main__":
    test_search()
