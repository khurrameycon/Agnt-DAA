"""
Improved Web Browsing Agent for sagax1
Enhanced with rate limiting handling and multiple search backends
"""

import os
import logging
import json
import time
import requests
from typing import Dict, Any, List, Optional, Callable
import random
from urllib.parse import quote_plus
import re

from app.agents.base_agent import BaseAgent


class MultiSearchTool:
    """Multi-backend search tool with rate limiting and fallbacks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.min_delay = 1.0  # Minimum delay between requests
        
        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
        ]
    
    def _rate_limit_delay(self):
        """Implement rate limiting delay"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            delay = self.min_delay - time_since_last + random.uniform(0.1, 0.5)
            self.logger.info(f"Rate limiting: waiting {delay:.2f} seconds")
            time.sleep(delay)
        
        self.last_request_time = time.time()
    
    def _get_headers(self):
        """Get randomized headers"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def search_duckduckgo_html(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search using DuckDuckGo HTML interface"""
        try:
            self._rate_limit_delay()
            
            # Use the regular DuckDuckGo search instead of lite
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            
            headers = self._get_headers()
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 202:
                self.logger.warning("DuckDuckGo rate limit hit, trying alternative approach")
                return []
            
            response.raise_for_status()
            html = response.text
            
            # Parse results using regex (simple but effective)
            results = []
            
            # Look for result links and titles
            link_pattern = r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>'
            
            links = re.findall(link_pattern, html)
            snippets = re.findall(snippet_pattern, html)
            
            for i, (url, title) in enumerate(links[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                
                # Clean up the data
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                url = url.replace('/l/?uddg=', '').replace('&rut=', '').split('&')[0]
                
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo HTML search failed: {e}")
            return []
    
    def search_searx(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search using SearX public instances"""
        searx_instances = [
            "https://searx.tiekoetter.com",
            "https://searx.be",
            "https://search.sapti.me",
            "https://searx.prvcy.eu",
        ]
        
        for instance in searx_instances:
            try:
                self._rate_limit_delay()
                
                search_url = f"{instance}/search"
                params = {
                    'q': query,
                    'format': 'json',
                    'categories': 'general'
                }
                
                headers = self._get_headers()
                
                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for item in data.get('results', [])[:max_results]:
                    results.append({
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'snippet': item.get('content', '')
                    })
                
                if results:
                    self.logger.info(f"Successfully used SearX instance: {instance}")
                    return results
                    
            except Exception as e:
                self.logger.warning(f"SearX instance {instance} failed: {e}")
                continue
        
        return []
    
    def search_bing_scrape(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Scrape Bing search results"""
        try:
            self._rate_limit_delay()
            
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
            headers = self._get_headers()
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            html = response.text
            results = []
            
            # Parse Bing results
            # This is a simplified parser - you might need to adjust the regex
            title_pattern = r'<h2><a href="([^"]+)"[^>]*>([^<]+)</a></h2>'
            snippet_pattern = r'<p class="b_lineclamp[^"]*">([^<]+)</p>'
            
            titles = re.findall(title_pattern, html)
            snippets = re.findall(snippet_pattern, html)
            
            for i, (url, title) in enumerate(titles[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                
                # Clean up the data
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Bing scraping failed: {e}")
            return []
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search with multiple backends and fallbacks"""
        self.logger.info(f"Searching for: {query}")
        
        # Try different search backends in order of preference
        search_methods = [
            ("DuckDuckGo HTML", self.search_duckduckgo_html),
            ("SearX", self.search_searx),
            ("Bing Scrape", self.search_bing_scrape),
        ]
        
        for method_name, search_method in search_methods:
            try:
                self.logger.info(f"Trying {method_name}...")
                results = search_method(query, max_results)
                
                if results:
                    self.logger.info(f"Successfully got {len(results)} results from {method_name}")
                    return results
                else:
                    self.logger.warning(f"{method_name} returned no results")
                    
            except Exception as e:
                self.logger.error(f"{method_name} failed: {e}")
                continue
        
        # If all methods fail, return a helpful message
        self.logger.error("All search methods failed")
        return [{
            'title': 'Search Temporarily Unavailable',
            'url': '',
            'snippet': 'All search services are currently unavailable due to rate limiting or connectivity issues. Please try again in a few minutes.'
        }]


class ImprovedWebBrowsingAgent(BaseAgent):
    """Improved web browsing agent with rate limiting and fallbacks"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the improved web browsing agent"""
        super().__init__(agent_id, config)
        
        # Get API provider and model configuration
        self.api_provider = config.get("api_provider", "groq")
        self.model_id = config.get("model_id", self._get_default_model())
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.1)
        
        # Initialize components
        self.search_tool = MultiSearchTool()
        self.api_provider_instance = None
        
        self.logger.info(f"Improved Web Browsing Agent initialized with {self.api_provider}")
    
    def _get_default_model(self):
        """Get default model based on API provider"""
        default_models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash-exp",
            "groq": "llama-3.3-70b-versatile",
            "anthropic": "claude-sonnet-4-20250514"
        }
        return default_models.get(self.api_provider, "llama-3.3-70b-versatile")
    
    def _initialize_api(self):
        """Initialize the API provider"""
        if self.api_provider_instance is not None:
            return
        
        from app.utils.api_providers import APIProviderFactory
        from app.core.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        
        # Get API key based on provider
        api_keys = {
            "openai": config_manager.get_openai_api_key(),
            "gemini": config_manager.get_gemini_api_key(),
            "groq": config_manager.get_groq_api_key(),
            "anthropic": config_manager.get_anthropic_api_key()
        }
        
        api_key = api_keys.get(self.api_provider)
        if not api_key:
            raise ValueError(f"{self.api_provider.upper()} API key is required")
        
        self.api_provider_instance = APIProviderFactory.create_provider(
            self.api_provider, api_key, self.model_id
        )
        self.logger.info(f"Initialized {self.api_provider.upper()} API")
    
    def search_and_summarize(self, query: str) -> str:
        """Search for information and create a summary"""
        try:
            self._initialize_api()
            
            # Perform search
            results = self.search_tool.search(query, max_results=5)
            
            if not results or not results[0].get('title'):
                return "I couldn't find any search results at the moment. Please try again later."
            
            # Format search results in a more compact way
            formatted_results = ""
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                url = result.get('url', '')
                snippet = result.get('snippet', 'No description available')
                
                # Truncate snippets to keep prompt manageable
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                
                formatted_results += f"{i}. {title}\n"
                formatted_results += f"   {snippet}\n"
                if url:
                    formatted_results += f"   Source: {url}\n"
                formatted_results += "\n"
            
            # Create a more concise prompt for Groq
            prompt = f"""Search Query: "{query}"

Search Results:
{formatted_results}

Please provide a clear and informative summary based on these search results. Include key points and mention relevant sources."""
            
            # Check prompt length and truncate if needed for Groq
            if len(prompt) > 2000:  # Conservative limit for Groq
                # Truncate the search results part
                truncated_results = formatted_results[:1000] + "\n[Results truncated...]"
                prompt = f"""Search Query: "{query}"

Search Results:
{truncated_results}

Please provide a clear and informative summary based on these search results."""
            
            # Generate summary with retry logic for Groq
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    messages = [{"content": prompt}]
                    summary = self.api_provider_instance.generate(
                        messages,
                        temperature=self.temperature,
                        max_tokens=min(self.max_tokens, 1024)  # Reduce max tokens for Groq
                    )
                    return summary
                    
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    if "503" in error_str or "service unavailable" in error_str:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Groq API 503 error, retrying in 2 seconds... (attempt {attempt + 1})")
                            time.sleep(2)  # Wait before retry
                            continue
                        else:
                            # If all retries failed, return formatted results without AI summary
                            self.logger.error("Groq API consistently unavailable, returning raw search results")
                            return self._format_raw_results(query, results)
                    else:
                        # For other errors, don't retry
                        raise api_error
            
        except Exception as e:
            self.logger.error(f"Error in search and summarize: {e}")
            # Return formatted search results as fallback
            if 'results' in locals() and results:
                return self._format_raw_results(query, results)
            else:
                return f"I encountered an error while searching: {str(e)}. Please try again."
    
    def _format_raw_results(self, query: str, results: List[Dict[str, str]]) -> str:
        """Format raw search results when AI summary fails"""
        formatted = f"# Search Results for: {query}\n\n"
        formatted += "*Note: AI summary temporarily unavailable, showing raw search results*\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            snippet = result.get('snippet', 'No description available')
            
            formatted += f"## {i}. {title}\n"
            if url:
                formatted += f"**URL:** {url}\n"
            formatted += f"**Description:** {snippet}\n\n"
        
        return formatted
    
    def run(self, input_text: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Run the agent with the given input"""
        try:
            if callback:
                callback("Searching the web...")
            
            # Clean up the input query
            query = input_text.strip()
            
            # Perform search and summarization
            result = self.search_and_summarize(query)
            
            if callback:
                callback("Search completed")
            
            # Add to history
            self.add_to_history(input_text, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error in web browsing agent: {str(e)}"
            self.logger.error(error_msg)
            return f"Sorry, I encountered an error while browsing the web: {error_msg}"
    
    def reset(self) -> None:
        """Reset the agent's state"""
        self.clear_history()
    
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities this agent has"""
        return [
            "web_search",
            "information_retrieval", 
            "content_summarization",
            "multi_backend_search",
            "rate_limit_handling",
            f"{self.api_provider}_api"
        ]