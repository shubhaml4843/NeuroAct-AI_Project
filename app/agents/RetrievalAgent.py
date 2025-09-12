"""Retrieval Agent for external data fetching and RAG"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from app.memory.rag import RAGPipeline
from app.config import get_search_apis, get_cache_config, get_rate_limit_config
from typing import Dict, Any, List, Optional
import requests
import json
import time
from pathlib import Path

logger = get_logger(__name__)

class RetrievalAgent:
    """Agent for retrieving external data and knowledge"""
    def __init__(self):
        self.name = "RetrievalAgent"
        self.tools = [
            "requests", "beautifulsoup4", "web_search", "document_processor",
            "feed_reader", "api_client", "data_validator", "cache_manager",
            "github_api", "huggingface_loader", "json_editor", "data_utils"
        ]
        self.rag_pipeline = RAGPipeline()
        self.session = requests.Session()
        
        # Load configurations
        self.search_apis = get_search_apis()
        self.cache_config = get_cache_config()
        self.rate_limit_config = get_rate_limit_config()
        
        # Initialize tools
        self._init_tools()
        
        # Rate limiting tracking
        self.rate_limits = {}
        
        logger.info(f"Initialized {self.name} with {len(self.tools)} tools")

    def _init_tools(self):
        """Initialize tool instances"""
        try:
            from app.tools.web_search import WebSearch
            from app.tools.cache_manager import CacheManager
            from app.tools.data_validator import DataValidator
            from app.tools.api_client import APIClient
            
            self.web_search = WebSearch()
            self.cache_manager = CacheManager(self.cache_config["cache_dir"])
            self.data_validator = DataValidator()
            self.api_client = APIClient()
            
        except ImportError as e:
            logger.warning(f"Some tools not available: {e}")
            # Fallback to basic functionality
            self.web_search = None
            self.cache_manager = None
            self.data_validator = None
            self.api_client = None

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute retrieval task"""
        try:
            query = task.inputs.get("query", "").lower()
            
            # Enhanced intent detection
            if "web" in query or "scrape" in query:
                return self._web_scraping(task.inputs)
            elif "api" in query or "fetch" in query:
                return self._api_fetch(task.inputs)
            elif "search" in query:
                return self._multi_search(task.inputs)
            elif "rag" in query or "knowledge" in query:
                return self._rag_search(task.inputs)
            elif "feed" in query or "rss" in query:
                return self._feed_retrieval(task.inputs)
            else:
                return self._auto_retrieve(task.inputs)
                
        except Exception as e:
            logger.error(f"RetrievalAgent execution failed: {str(e)}")
            return {"status": "error", "errors": [str(e)], "agent": self.name}

    def _web_scraping(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced web scraping with validation and caching"""
        try:
            url = inputs.get("url", "")
            if not url:
                return {"status": "error", "errors": ["No URL provided"]}
            
            # Validate URL
            if self.data_validator and not self.data_validator.validate_url(url)["is_valid"]:
                return {"status": "error", "errors": ["Invalid URL provided"]}
            
            # Check cache first
            cache_key = f"scrape_{hash(url)}"
            if self.cache_manager:
                cached_result = self.cache_manager.get_cache(cache_key)
                if cached_result:
                    logger.info(f"Retrieved cached content for {url}")
                    return cached_result
            
            # Rate limiting
            if not self._check_rate_limit("scraping"):
                return {"status": "error", "errors": ["Rate limit exceeded"]}
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Enhanced content extraction
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "aside"]):
                    element.decompose()
                
                # Extract structured data
                title = soup.title.string if soup.title else "No title"
                text = soup.get_text()
                
                # Clean text more thoroughly
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 3)
                
                # Extract metadata
                meta_description = ""
                meta_tag = soup.find("meta", attrs={"name": "description"})
                if meta_tag:
                    meta_description = meta_tag.get("content", "")
                
                result = {
                    "status": "success",
                    "action": "web_scraping",
                    "output": clean_text[:3000],  # Increased limit
                    "metadata": {
                        "url": url,
                        "title": title,
                        "description": meta_description,
                        "content_length": len(clean_text),
                        "word_count": len(clean_text.split()),
                        "cached": False
                    },
                    "message": f"Successfully scraped {len(clean_text)} characters from {url}"
                }
                
                # Cache the result
                if self.cache_manager:
                    self.cache_manager.set_cache(cache_key, result, self.cache_config["ttl"])
                
                return result
                
            except ImportError:
                # Fallback without BeautifulSoup
                result = {
                    "status": "success",
                    "action": "web_scraping",
                    "output": response.text[:2000],
                    "metadata": {"url": url, "raw_content": True, "cached": False},
                    "message": f"Raw content retrieved from {url}"
                }
                
                if self.cache_manager:
                    self.cache_manager.set_cache(cache_key, result, self.cache_config["ttl"])
                
                return result
                
        except Exception as e:
            logger.error(f"Web scraping failed for {url}: {str(e)}")
            return {"status": "error", "errors": [f"Web scraping failed: {str(e)}"]}

    def _api_fetch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced API fetching with better error handling"""
        try:
            url = inputs.get("url", inputs.get("api_url", ""))
            method = inputs.get("method", "GET").upper()
            headers = inputs.get("headers", {})
            params = inputs.get("params", {})
            data = inputs.get("data", {})

            if not url:
                return {"status": "error", "errors": ["No API URL provided"]}

            # Validate URL
            if self.data_validator and not self.data_validator.validate_url(url)["is_valid"]:
                return {"status": "error", "errors": ["Invalid API URL"]}

            # Check cache
            cache_key = f"api_{hash(url + str(params) + str(data))}"
            if self.cache_manager and method == "GET":
                cached_result = self.cache_manager.get_cache(cache_key)
                if cached_result:
                    return cached_result

            # Rate limiting
            if not self._check_rate_limit("api"):
                return {"status": "error", "errors": ["API rate limit exceeded"]}

            # Make API request
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params, timeout=15)
            elif method == "POST":
                response = self.session.post(url, headers=headers, json=data, timeout=15)
            else:
                return {"status": "error", "errors": [f"Unsupported method: {method}"]}
            
            response.raise_for_status()

            # Try to parse JSON
            try:
                json_data = response.json()
                result = {
                    "status": "success",
                    "action": "api_fetch",
                    "output": json_data,
                    "metadata": {
                        "url": url,
                        "method": method,
                        "status_code": response.status_code,
                        "content_type": response.headers.get('content-type', 'unknown'),
                        "response_size": len(response.text),
                        "cached": False
                    },
                    "message": f"Successfully fetched data from API: {response.status_code}"
                }
            except json.JSONDecodeError:
                result = {
                    "status": "success",
                    "action": "api_fetch",
                    "output": response.text,
                    "metadata": {
                        "url": url, 
                        "raw_response": True,
                        "status_code": response.status_code,
                        "cached": False
                    },
                    "message": "API response retrieved (non-JSON)"
                }

            # Cache GET requests
            if self.cache_manager and method == "GET":
                self.cache_manager.set_cache(cache_key, result, self.cache_config["ttl"])

            return result
                
        except Exception as e:
            logger.error(f"API fetch failed: {str(e)}")
            return {"status": "error", "errors": [f"API fetch failed: {str(e)}"]}

    def _multi_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-source search using configured APIs"""
        try:
            query = inputs.get("query", inputs.get("search_query", ""))
            sources = inputs.get("sources", ["duckduckgo", "wikipedia"])
            max_results = inputs.get("max_results", 5)

            if not query:
                return {"status": "error", "errors": ["No search query provided"]}

            results = {}
            
            # DuckDuckGo search
            if "duckduckgo" in sources:
                try:
                    search_url = f"{self.search_apis['duckduckgo']}?q={query}&format=json&no_html=1"
                    response = self.session.get(search_url, timeout=10)
                    if response.status_code == 200:
                        results["duckduckgo"] = response.json()
                except Exception as e:
                    results["duckduckgo"] = {"error": str(e)}

            # Wikipedia search
            if "wikipedia" in sources:
                try:
                    wiki_url = f"{self.search_apis['wikipedia']}page/summary/{query}"
                    response = self.session.get(wiki_url, timeout=10)
                    if response.status_code == 200:
                        results["wikipedia"] = response.json()
                except Exception as e:
                    results["wikipedia"] = {"error": str(e)}

            # ArXiv search
            if "arxiv" in sources:
                try:
                    arxiv_url = f"{self.search_apis['arxiv']}?search_query=all:{query}&max_results={max_results}"
                    response = self.session.get(arxiv_url, timeout=10)
                    if response.status_code == 200:
                        results["arxiv"] = response.text  # ArXiv returns XML
                except Exception as e:
                    results["arxiv"] = {"error": str(e)}

            return {
                "status": "success",
                "action": "multi_search",
                "output": results,
                "metadata": {
                    "query": query,
                    "sources_searched": list(results.keys()),
                    "max_results": max_results
                },
                "message": f"Multi-source search completed for '{query}'"
            }

        except Exception as e:
            return {"status": "error", "errors": [f"Multi-search failed: {str(e)}"]}

    def _rag_search(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced RAG search with metadata"""
        try:
            query = inputs.get("query", inputs.get("search_query", ""))
            k = inputs.get("k", 5)

            if not query:
                return {"status": "error", "errors": ["No search query provided"]}
            
            # Search in RAG pipeline
            results = self.rag_pipeline.retrieve_context(query, k)
            
            return {
                "status": "success",
                "action": "rag_search",
                "output": results,
                "metadata": {
                    "query": query,
                    "results_count": len(results),
                    "k": k,
                    "search_type": "vector_similarity"
                },
                "message": f"Found {len(results)} relevant documents in knowledge base"
            }
        except Exception as e:
            return {"status": "error", "errors": [f"RAG search failed: {str(e)}"]}

    def _feed_retrieval(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """RSS/Atom feed retrieval"""
        try:
            feed_url = inputs.get("url", inputs.get("feed_url", ""))
            if not feed_url:
                return {"status": "error", "errors": ["No feed URL provided"]}

            # Basic RSS parsing (can be enhanced with feedparser)
            response = self.session.get(feed_url, timeout=10)
            response.raise_for_status()

            return {
                "status": "success",
                "action": "feed_retrieval",
                "output": response.text[:2000],
                "metadata": {
                    "feed_url": feed_url,
                    "content_type": response.headers.get('content-type', 'unknown')
                },
                "message": f"Successfully retrieved feed from {feed_url}"
            }

        except Exception as e:
            return {"status": "error", "errors": [f"Feed retrieval failed: {str(e)}"]}

    def _auto_retrieve(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced auto-retrieval with better detection"""
        try:
            query = inputs.get("query", "")
            
            # Check if URL is provided
            if inputs.get("url"):
                if "rss" in inputs.get("url", "") or "feed" in inputs.get("url", ""):
                    return self._feed_retrieval(inputs)
                else:
                    return self._web_scraping(inputs)
            
            # Check if it's an API request
            if "api" in query or inputs.get("api_url"):
                return self._api_fetch(inputs)
            
            # Check if it's a search query
            if any(word in query for word in ["search", "find", "lookup"]):
                return self._multi_search(inputs)
            
            # Default to RAG search
            return self._rag_search(inputs)
            
        except Exception as e:
            return {"status": "error", "errors": [f"Auto retrieval failed: {str(e)}"]}

    def _check_rate_limit(self, operation: str) -> bool:
        """Check if operation is within rate limits"""
        current_time = time.time()
        window = self.rate_limit_config["window"]
        max_requests = self.rate_limit_config["requests"]
        
        if operation not in self.rate_limits:
            self.rate_limits[operation] = []
        
        # Remove old requests outside the window
        self.rate_limits[operation] = [
            req_time for req_time in self.rate_limits[operation]
            if current_time - req_time < window
        ]
        
        # Check if we're within limits
        if len(self.rate_limits[operation]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[operation].append(current_time)
        return True

    def add_to_knowledge_base(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add text to RAG knowledge base with validation"""
        try:
            # Validate text
            if self.data_validator:
                clean_text = self.data_validator.sanitize_text(text)
            else:
                clean_text = text
            
            self.rag_pipeline.add_knowledge(clean_text, metadata or {})
            
            return {
                "status": "success",
                "message": "Added to knowledge base",
                "metadata": {
                    "text_length": len(clean_text),
                    "original_length": len(text),
                    "sanitized": len(clean_text) != len(text)
                }
            }
        except Exception as e:
            return {"status": "error", "errors": [f"Failed to add to knowledge base: {str(e)}"]}

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval agent statistics"""
        try:
            cache_stats = {}
            if self.cache_manager:
                cache_stats = self.cache_manager.get_cache_stats()
            
            return {
                "status": "success",
                "stats": {
                    "tools_available": len([tool for tool in [self.web_search, self.cache_manager, self.data_validator, self.api_client] if tool]),
                    "rate_limits": {op: len(reqs) for op, reqs in self.rate_limits.items()},
                    "cache_stats": cache_stats,
                    "search_apis": list(self.search_apis.keys())
                }
            }
        except Exception as e:
            return {"status": "error", "errors": [f"Failed to get stats: {str(e)}"]}