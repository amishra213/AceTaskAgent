"""
Web Search Sub-Agent for TaskManager with Dynamic Content Support

This agent handles web search and content retrieval operations.
Supports multiple search engines, static scraping, and dynamic JavaScript-heavy sites via Playwright.
Integrates with Vision agent for screenshot analysis.
Includes deep search capability for human-like browsing and information extraction.

Migration Status: Week 7 Day 2 - Dual Format Support
- Supports both legacy dict and standardized AgentExecutionRequest/Response
- Maintains 100% backward compatibility
- Publishes SystemEvent on completion for event-driven workflows
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import json
import asyncio
import tempfile
import re
import os
import time

# Import standardized schemas and utilities (Week 1-2 implementation)
from task_manager.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    create_system_event,
    create_error_response
)
from task_manager.utils import (
    auto_convert_response,
    validate_agent_execution_response,
    exception_to_error_response,
    InvalidParameterError,
    InvalidOperationError,
    WebSearchError,
    ScrapingError,
    MissingDependencyError,
    NetworkError,
    wrap_exception
)
from task_manager.core.event_bus import get_event_bus

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """Agent for handling web search, static scraping, and dynamic web interaction."""
    
    def __init__(self):
        """Initialize the Web Search Agent with dual-format support."""
        self.agent_name = "web_search_agent"
        self.supported_operations = [
            "search",
            "scrape",
            "fetch",
            "summarize",
            "smart_scrape",
            "capture_screenshot",
            "handle_pagination",
            "deep_search",
            "research"
        ]
        
        # Initialize event bus for event-driven workflows
        self.event_bus = get_event_bus()
        
        self.search_lib = self._check_dependencies()
        self.playwright_available = self._check_playwright()
        
        # Load search engine configuration from environment
        # Available backends: auto, duckduckgo, google, brave, yahoo, yandex, mojeek, grokipedia, wikipedia
        self.default_backend = os.getenv('WEBSEARCH_BACKEND', 'auto')
        self.default_region = os.getenv('WEBSEARCH_REGION', 'wt-wt')
        self.search_timeout = int(os.getenv('WEBSEARCH_TIMEOUT', '10'))
        
        # Custom search engine URL (if provided - for future use)
        self.custom_search_url = os.getenv('WEBSEARCH_CUSTOM_URL', None)
        
        logger.info(f"Web Search Agent initialized with dual-format support (backend={self.default_backend}, region={self.default_region})")
    
    def _check_dependencies(self) -> Optional[str]:
        """
        Check which search libraries are available.
        
        Returns:
            Library name or None if no library available
        """
        try:
            from ddgs import DDGS  # noqa: F401
            return "ddgs"
        except ImportError:
            pass
        
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            return "duckduckgo"
        except ImportError:
            pass
        
        try:
            import requests  # noqa: F401
            return "requests"
        except ImportError:
            pass
        
        logger.warning("No search libraries installed. Install duckduckgo-search and requests")
        return None
    
    def _check_playwright(self) -> bool:
        """
        Check if Playwright is available for dynamic content rendering.
        
        Returns:
            True if Playwright is available, False otherwise
        """
        try:
            import playwright  # type: ignore  # noqa: F401
            logger.debug("Playwright available for dynamic content rendering")
            return True
        except ImportError:
            logger.debug("Playwright not installed. Install with: pip install playwright")
            return False
    
    def _reformulate_query(self, original_query: str, attempt: int) -> str:
        """
        Reformulate search query for retry attempts.
        
        Args:
            original_query: The original search query
            attempt: Retry attempt number (1-based)
        
        Returns:
            Reformulated query string
        """
        strategies = [
            # Strategy 1: Remove quotes and special characters
            lambda q: re.sub(r'["\']', '', q),
            # Strategy 2: Simplify - remove common words
            lambda q: ' '.join([word for word in q.split() if word.lower() not in 
                               ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'list', 'of']]),
            # Strategy 3: Add contextual words
            lambda q: f"{q} information details",
            # Strategy 4: Make it more general (remove specific terms)
            lambda q: ' '.join(q.split()[:3]),  # Take first 3 words
            # Strategy 5: Add location/official context
            lambda q: f"official {q} site",
        ]
        
        if attempt <= len(strategies):
            reformulated = strategies[attempt - 1](original_query)
            logger.info(f"Query reformulation (attempt {attempt}): '{original_query}' -> '{reformulated}'")
            return reformulated
        else:
            # Fallback: use original query
            return original_query

    def search(
        self,
        query: str,
        max_results: int = 10,
        language: str = "en",
        source: Optional[str] = None,
        retry_on_empty: bool = True,
        max_retries: int = 3,
        backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a web search with automatic retry on zero results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            language: Language/region code for search results (e.g., 'wt-wt', 'us', 'uk')
            source: Optional specific search engine (deprecated, use backend instead)
            retry_on_empty: Whether to retry with reformulated query if no results found
            max_retries: Maximum number of retry attempts
            backend: Search engine to use. Options:
                - 'auto' (default): Automatically select best available engine
                - 'duckduckgo': DuckDuckGo search
                - 'google': Google search
                - 'brave': Brave search  
                - 'yahoo': Yahoo search
                - 'yandex': Yandex search
                - 'mojeek': Mojeek search
                - 'wikipedia': Wikipedia search
                - 'grokipedia': Grokipedia search
        
        Returns:
            Dictionary with search results
        """
        try:
            if not self.search_lib:
                return {
                    "success": False,
                    "error": "No search library installed",
                    "query": query
                }
            
            # Use provided backend or fall back to default from env
            search_backend = backend or self.default_backend
            search_region = language or self.default_region
            
            original_query = query
            attempt = 0
            all_attempts = []
            
            while attempt <= max_retries:
                current_query = query if attempt == 0 else self._reformulate_query(original_query, attempt)
                attempt += 1
                
                logger.info(f"Searching web (attempt {attempt}/{max_retries + 1}): {current_query}")
                logger.debug(f"Using backend='{search_backend}', region='{search_region}'")
                
                # Use DuckDuckGo as primary search engine (prefer new 'ddgs' package)
                try:
                    from ddgs import DDGS  # noqa: F401
                    
                    ddgs = DDGS()
                    # Pass backend parameter to DDGS
                    results = ddgs.text(
                        current_query, 
                        max_results=max_results, 
                        region=search_region,
                        backend=search_backend
                    )
                    
                    formatted_results = []
                    for idx, result in enumerate(results, 1):
                        formatted_results.append({
                            "rank": idx,
                            "title": result.get('title', ''),
                            "url": result.get('href', ''),
                            "snippet": result.get('body', ''),
                            "source": result.get('source', 'Web'),
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    logger.info(f"Found {len(formatted_results)} results for query")
                    
                    # Track this attempt
                    all_attempts.append({
                        "attempt": attempt,
                        "query": current_query,
                        "results_count": len(formatted_results)
                    })
                    
                    # If we got results, return success
                    if len(formatted_results) > 0:
                        return {
                            "success": True,
                            "query": current_query,
                            "original_query": original_query,
                            "results_count": len(formatted_results),
                            "results": formatted_results,
                            "search_time": datetime.now().isoformat(),
                            "attempts": all_attempts,
                            "retry_count": attempt - 1
                        }
                    
                    # If no results and we shouldn't retry, return failure
                    if not retry_on_empty or attempt > max_retries:
                        logger.warning(f"No results found after {attempt} attempts")
                        return {
                            "success": False,
                            "error": f"No results found after {attempt} attempts with various query formulations",
                            "query": current_query,
                            "original_query": original_query,
                            "results_count": 0,
                            "results": [],
                            "search_time": datetime.now().isoformat(),
                            "attempts": all_attempts,
                            "retry_count": attempt - 1
                        }
                    
                    # Continue to next retry
                    logger.info(f"0 results found, will retry with reformulated query...")
                
                except ImportError:
                    # Fallback to old duckduckgo_search package
                    try:
                        from duckduckgo_search import DDGS  # noqa: F401
                        
                        ddgs = DDGS()
                        # Note: Old package may not support backend parameter
                        try:
                            results = ddgs.text(
                                current_query, 
                                max_results=max_results, 
                                region=search_region,
                                backend=search_backend
                            )
                        except TypeError:
                            # Old version doesn't support backend parameter
                            logger.debug("Old DDGS package doesn't support backend parameter")
                            results = ddgs.text(current_query, max_results=max_results, region=search_region)
                        
                        formatted_results = []
                        for idx, result in enumerate(results, 1):
                            formatted_results.append({
                                "rank": idx,
                                "title": result.get('title', ''),
                                "url": result.get('href', ''),
                                "snippet": result.get('body', ''),
                                "source": result.get('source', 'Web'),
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        logger.info(f"Found {len(formatted_results)} results for query")
                        
                        # Track this attempt
                        all_attempts.append({
                            "attempt": attempt,
                            "query": current_query,
                            "results_count": len(formatted_results)
                        })
                        
                        # If we got results, return success
                        if len(formatted_results) > 0:
                            return {
                                "success": True,
                                "query": current_query,
                                "original_query": original_query,
                                "results_count": len(formatted_results),
                                "results": formatted_results,
                                "search_time": datetime.now().isoformat(),
                                "attempts": all_attempts,
                                "retry_count": attempt - 1
                            }
                        
                        # If no results and we shouldn't retry, return failure
                        if not retry_on_empty or attempt > max_retries:
                            logger.warning(f"No results found after {attempt} attempts")
                            return {
                                "success": False,
                                "error": f"No results found after {attempt} attempts with various query formulations",
                                "query": current_query,
                                "original_query": original_query,
                                "results_count": 0,
                                "results": [],
                                "search_time": datetime.now().isoformat(),
                                "attempts": all_attempts,
                                "retry_count": attempt - 1
                            }
                        
                        # Continue to next retry
                        logger.info(f"0 results found, will retry with reformulated query...")
                    
                    except ImportError:
                        logger.debug("Neither ddgs nor duckduckgo_search library available, using fallback method")
                        return {
                            "success": False,
                            "error": "ddgs not installed. Install with: pip install ddgs (or legacy: pip install duckduckgo-search)",
                            "query": query
                        }
            
            # Should not reach here, but just in case
            return {
                "success": False,
                "error": "Max retries exceeded with no results",
                "query": query,
                "original_query": original_query,
                "results_count": 0,
                "results": [],
                "attempts": all_attempts
            }
        
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def scrape(
        self,
        url: str,
        extract_text: bool = True,
        extract_links: bool = True,
        extract_images: bool = False
    ) -> Dict[str, Any]:
        """
        Scrape static content from a URL using BeautifulSoup.
        
        Args:
            url: URL to scrape
            extract_text: Whether to extract text content
            extract_links: Whether to extract links
            extract_images: Whether to extract image URLs
        
        Returns:
            Dictionary with scraped content
        """
        try:
            import requests  # noqa: F401
            from bs4 import BeautifulSoup  # noqa: F401
            
            logger.info(f"Scraping static content from: {url}")
            
            # Fetch page with timeout and user agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = soup.find('title')
            description = soup.find('meta', {'name': 'description'})
            
            scraped_data: Dict[str, Any] = {
                "success": True,
                "url": url,
                "title": title.string if title else "",
                "description": description.get('content', '') if description else "",
                "status_code": response.status_code,
                "content_type": response.headers.get('content-type', ''),
                "raw_html": response.text  # Include raw HTML for structured extraction
            }
            
            # Extract text
            if extract_text:
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                scraped_data["text_content"] = text[:5000]  # Limit to 5000 chars
                scraped_data["text_length"] = len(text)
            
            # Extract links
            if extract_links:
                links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    link_text = link.get_text(strip=True)
                    if href and link_text:
                        links.append({
                            "text": link_text[:100],
                            "url": href
                        })
                
                scraped_data["links"] = links[:20]  # Limit to 20 links
                scraped_data["links_count"] = len(links)
            
            # Extract images
            if extract_images:
                images = []
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    alt = img.get('alt', '')
                    if src:
                        images.append({
                            "src": src,
                            "alt": alt[:100] if alt else ""
                        })
                
                scraped_data["images"] = images[:10]  # Limit to 10 images
                scraped_data["images_count"] = len(images)
            
            scraped_data["scraped_at"] = datetime.now().isoformat()
            
            logger.info(f"Successfully scraped {url}")
            return scraped_data
        
        except Exception as e:
            logger.error(f"Error scraping URL: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def smart_scrape(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        handle_pagination: bool = False,
        max_pages: int = 3,
        scroll_pause_time: float = 1.0
    ) -> Dict[str, Any]:
        """
        Dynamically scrape JavaScript-heavy sites using Playwright.
        
        This method launches a headless browser, renders JavaScript, and extracts content.
        Can handle paginated content and 'Load More' buttons.
        
        Args:
            url: URL to scrape
            wait_selector: CSS selector to wait for before extraction
            handle_pagination: Whether to follow pagination/Load More buttons
            max_pages: Maximum number of pages to scrape if handling pagination
            scroll_pause_time: Time to pause after scrolling (for content loading)
        
        Returns:
            Dictionary with scraped dynamic content
        """
        if not self.playwright_available:
            return {
                "success": False,
                "error": "Playwright not installed. Install with: pip install playwright",
                "url": url,
                "note": "Falling back to static scrape"
            }
        
        try:
            logger.info(f"Smart scraping JavaScript-heavy content from: {url}")
            
            # Run async Playwright operations
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self._async_smart_scrape(url, wait_selector, handle_pagination, max_pages, scroll_pause_time)
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error in smart scrape: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def _async_smart_scrape(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        handle_pagination: bool = False,
        max_pages: int = 3,
        scroll_pause_time: float = 1.0
    ) -> Dict[str, Any]:
        """
        Async helper for smart scraping using Playwright.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore
            from bs4 import BeautifulSoup  # noqa: F401
            import asyncio
            
            async with async_playwright() as p:
                # Launch browser in headless mode
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate to URL with wait until network idle
                await page.goto(url, wait_until="networkidle")
                
                # Wait for specific selector if provided
                if wait_selector:
                    try:
                        await page.wait_for_selector(wait_selector, timeout=5000)
                    except Exception as e:
                        logger.warning(f"Selector {wait_selector} not found: {str(e)}")
                
                # Collect content from all pages
                all_content = []
                page_count = 0
                
                while page_count < max_pages:
                    page_count += 1
                    
                    # Scroll to load lazy-loaded content
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(scroll_pause_time)
                    
                    # Get page content
                    content = await page.content()
                    all_content.append(content)
                    
                    # Check for pagination or Load More button
                    if handle_pagination and page_count < max_pages:
                        # Try common "Load More" button selectors
                        load_more_selectors = [
                            'button:has-text("Load More")',
                            'button:has-text("load more")',
                            'button[class*="load-more"]',
                            'button[class*="loadMore"]',
                            'a[class*="load-more"]',
                            'a[class*="next"]'
                        ]
                        
                        found_button = False
                        for selector in load_more_selectors:
                            try:
                                button = page.locator(selector).first
                                if await button.is_visible():
                                    await button.click()
                                    await asyncio.sleep(scroll_pause_time)
                                    found_button = True
                                    logger.info(f"Clicked Load More button, loading page {page_count + 1}")
                                    break
                            except Exception:
                                continue
                        
                        if not found_button:
                            logger.info("No more Load More buttons found")
                            break
                    else:
                        break
                
                # Parse combined content
                combined_html = "".join(all_content)
                soup = BeautifulSoup(combined_html, 'html.parser')
                
                # Extract text
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Get title
                title = soup.find('title')
                
                await browser.close()
                
                logger.info(f"Successfully smart scraped {url} ({page_count} pages)")
                
                return {
                    "success": True,
                    "url": url,
                    "title": title.string if title else "",
                    "text_content": text[:8000],  # Larger limit for dynamic content
                    "text_length": len(text),
                    "pages_scraped": page_count,
                    "scraped_at": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error in async smart scrape: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def capture_screenshot(
        self,
        url: str,
        selector: Optional[str] = None,
        output_dir: Optional[str] = None,
        wait_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Capture screenshot of a webpage or specific element for Vision agent analysis.
        
        Args:
            url: URL to screenshot
            selector: CSS selector for specific element (if None, captures full page)
            output_dir: Directory to save screenshot
            wait_selector: Selector to wait for before capturing
        
        Returns:
            Dictionary with screenshot path and metadata
        """
        if not self.playwright_available:
            return {
                "success": False,
                "error": "Playwright not installed. Install with: pip install playwright",
                "url": url
            }
        
        try:
            logger.info(f"Capturing screenshot from: {url}")
            
            # Create output directory if not provided
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="web_screenshots_")
            else:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self._async_capture_screenshot(url, selector, output_dir, wait_selector)
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error capturing screenshot: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def _async_capture_screenshot(
        self,
        url: str,
        selector: Optional[str] = None,
        output_dir: str = "",
        wait_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async helper for screenshot capture using Playwright.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore
            import asyncio
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate and wait
                await page.goto(url, wait_until="networkidle")
                
                # Wait for specific selector if provided
                if wait_selector:
                    try:
                        await page.wait_for_selector(wait_selector, timeout=5000)
                    except Exception as e:
                        logger.warning(f"Selector {wait_selector} not found: {str(e)}")
                
                await asyncio.sleep(1)  # Wait for rendering
                
                # Generate screenshot filename
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.replace('.', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_name = f"screenshot_{domain}_{timestamp}.png"
                screenshot_path = str(Path(output_dir) / screenshot_name)
                
                # Capture screenshot
                if selector:
                    # Capture specific element
                    try:
                        element = page.locator(selector).first
                        await element.screenshot(path=screenshot_path)
                        logger.info(f"Captured element screenshot: {screenshot_path}")
                    except Exception as e:
                        logger.warning(f"Failed to capture element, capturing full page: {str(e)}")
                        await page.screenshot(path=screenshot_path)
                else:
                    # Capture full page
                    await page.screenshot(path=screenshot_path, full_page=True)
                
                await browser.close()
                
                return {
                    "success": True,
                    "url": url,
                    "screenshot_path": screenshot_path,
                    "selector": selector,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error in async screenshot capture: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def handle_pagination(
        self,
        url: str,
        next_button_selector: str,
        content_selector: str,
        max_pages: int = 5
    ) -> Dict[str, Any]:
        """
        Handle paginated content extraction by following next page buttons.
        
        Args:
            url: Starting URL
            next_button_selector: CSS selector for next page button
            content_selector: CSS selector for content to extract
            max_pages: Maximum pages to scrape
        
        Returns:
            Dictionary with all page contents
        """
        if not self.playwright_available:
            return {
                "success": False,
                "error": "Playwright not installed",
                "url": url
            }
        
        try:
            logger.info(f"Handling pagination for: {url}")
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self._async_handle_pagination(url, next_button_selector, content_selector, max_pages)
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error handling pagination: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def _async_handle_pagination(
        self,
        url: str,
        next_button_selector: str,
        content_selector: str,
        max_pages: int = 5
    ) -> Dict[str, Any]:
        """
        Async helper for pagination handling using Playwright.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore
            from bs4 import BeautifulSoup  # noqa: F401
            import asyncio
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                all_pages_content = []
                current_url = url
                page_count = 0
                
                while page_count < max_pages:
                    page_count += 1
                    
                    # Navigate to current page
                    await page.goto(current_url, wait_until="networkidle")
                    await asyncio.sleep(1)
                    
                    # Extract content
                    try:
                        content_element = page.locator(content_selector).first
                        content_html = await content_element.inner_html()
                        all_pages_content.append({
                            "page": page_count,
                            "url": current_url,
                            "content": content_html[:2000]
                        })
                    except Exception as e:
                        logger.warning(f"Could not extract content with selector: {str(e)}")
                    
                    # Check for next button
                    try:
                        next_button = page.locator(next_button_selector).first
                        if await next_button.is_enabled():
                            # Get next page URL or click button
                            href = await next_button.get_attribute("href")
                            if href:
                                from urllib.parse import urljoin
                                current_url = urljoin(current_url, href)
                            else:
                                await next_button.click()
                                await asyncio.sleep(2)
                                current_url = page.url
                        else:
                            logger.info(f"Next button disabled or not found, stopping at page {page_count}")
                            break
                    except Exception as e:
                        logger.info(f"No next button found: {str(e)}, stopping at page {page_count}")
                        break
                
                await browser.close()
                
                logger.info(f"Successfully paginated {page_count} pages")
                
                return {
                    "success": True,
                    "start_url": url,
                    "pages_scraped": page_count,
                    "pages_content": all_pages_content,
                    "scraped_at": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error in async pagination: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def fetch(
        self,
        url: str,
        save_to_file: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch raw content from a URL.
        
        Args:
            url: URL to fetch
            save_to_file: Optional path to save content to file
            headers: Optional custom headers
        
        Returns:
            Dictionary with fetched content
        """
        try:
            import requests  # noqa: F401
            
            logger.info(f"Fetching content from: {url}")
            
            default_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            if headers:
                default_headers.update(headers)
            
            response = requests.get(url, headers=default_headers, timeout=10)
            response.raise_for_status()
            
            fetch_data: Dict[str, Any] = {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "content_length": len(response.content),
                "content_type": response.headers.get('content-type', ''),
                "headers": dict(response.headers),
                "encoding": response.encoding
            }
            
            # Try to extract text
            try:
                fetch_data["content"] = response.text[:5000]
            except Exception:
                fetch_data["content"] = "[Binary content - not text]"
            
            # Save to file if requested
            if save_to_file:
                Path(save_to_file).parent.mkdir(parents=True, exist_ok=True)
                with open(save_to_file, 'wb') as f:
                    f.write(response.content)
                fetch_data["saved_to_file"] = save_to_file
            
            fetch_data["fetched_at"] = datetime.now().isoformat()
            
            logger.info(f"Successfully fetched {url}")
            return fetch_data
        
        except Exception as e:
            logger.error(f"Error fetching URL: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def summarize(
        self,
        url: str,
        max_sentences: int = 5
    ) -> Dict[str, Any]:
        """
        Fetch and summarize content from a URL.
        
        Args:
            url: URL to summarize
            max_sentences: Maximum sentences in summary
        
        Returns:
            Dictionary with summarized content
        """
        try:
            import requests  # noqa: F401
            from bs4 import BeautifulSoup  # noqa: F401
            
            logger.info(f"Summarizing content from: {url}")
            
            # First scrape the content
            scrape_result = self.scrape(url, extract_text=True, extract_links=False)
            
            if not scrape_result.get("success"):
                return scrape_result
            
            text_content = scrape_result.get("text_content", "")
            
            if not text_content:
                return {
                    "success": False,
                    "error": "No text content to summarize",
                    "url": url
                }
            
            # Simple summarization by extracting key sentences
            sentences = text_content.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            # Take first N sentences as summary (simple approach)
            summary_sentences = sentences[:max_sentences]
            summary = '. '.join(summary_sentences)
            
            if summary and not summary.endswith('.'):
                summary += '.'
            
            logger.info(f"Generated summary for {url}")
            
            return {
                "success": True,
                "url": url,
                "title": scrape_result.get("title", ""),
                "summary": summary,
                "sentence_count": len(summary_sentences),
                "original_length": len(text_content),
                "summary_length": len(summary),
                "summarized_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error summarizing URL: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def _detect_and_switch_language(self, url: str, html_content: str) -> Optional[str]:
        """
        Detect if page is in a non-English language and find English version.
        
        Args:
            url: Current page URL
            html_content: HTML content of the page
        
        Returns:
            English version URL if found, None otherwise
        """
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode
            
            # Quick check: if URL already contains English indicators, skip detection
            url_lower = url.lower()
            if any(indicator in url_lower for indicator in ['/en/', '/en', '/english/', '/english']):
                # Already on English version (likely)
                return None
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for language switcher links with English text
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()
                
                # Check for English language switcher
                if text in ['english', 'en', 'en-us', 'en-gb'] or 'english' in text:
                    if '/en' in str(href).lower() or '/english' in str(href).lower():
                        english_url = urljoin(url, str(href))
                        if english_url != url and '/en' in english_url.lower():
                            logger.info(f"Found English language switcher: {english_url}")
                            return english_url
            
            # Check URL patterns - if URL contains language code, try English variants
            parsed = urlparse(url)
            path = parsed.path
            
            # Common non-English language codes in Indian government sites
            non_english_codes = ['/kn', '/ka', '/hi', '/ta', '/te', '/mr', '/gu', '/bn', 
                               '/kn/', '/ka/', '/hi/', '/ta/', '/te/', '/mr/', '/gu/', '/bn/']
            
            for code in non_english_codes:
                if code in path:
                    # Try replacing with /en
                    new_path = path.replace(code, '/en', 1)
                    new_url = urlunparse((
                        parsed.scheme,
                        parsed.netloc,
                        new_path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment
                    ))
                    
                    if new_url != url:
                        logger.info(f"Trying English URL variant: {new_url}")
                        return new_url
            
            # Check query parameters for language
            if parsed.query:
                query_params = parse_qs(parsed.query)
                lang_params = ['lang', 'language', 'locale', 'hl']
                
                for param in lang_params:
                    if param in query_params and query_params[param][0] not in ['en', 'english', 'en-us']:
                        # Create new query with English language
                        new_params = query_params.copy()
                        new_params[param] = ['en']
                        new_query = urlencode(new_params, doseq=True)
                        new_url = urlunparse((
                            parsed.scheme,
                            parsed.netloc,
                            parsed.path,
                            parsed.params,
                            new_query,
                            parsed.fragment
                        ))
                        
                        if new_url != url:
                            logger.info(f"Trying English query parameter: {new_url}")
                            return new_url
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting language: {str(e)}")
            return None
    
    def deep_search(
        self,
        query: str,
        target_url: Optional[str] = None,
        max_results: int = 5,
        max_depth: int = 2,
        relevance_keywords: Optional[List[str]] = None,
        extract_structured_data: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a deep search like a human would - traverse search results,
        visit each page, extract relevant information, and follow internal links.
        
        This method mimics human browsing behavior:
        1. If target_url provided, starts there; otherwise searches with query
        2. Visits each result page and extracts content
        3. Evaluates content relevance based on keywords
        4. Follows promising internal links (up to max_depth)
        5. Aggregates all findings into a comprehensive result
        
        Args:
            query: Search query or topic to research
            target_url: Optional specific URL to start from (skip search)
            max_results: Maximum search results to visit (default 5)
            max_depth: How deep to follow internal links (default 2)
            relevance_keywords: Keywords to filter relevant content
            extract_structured_data: Whether to try extracting lists/tables
        
        Returns:
            Dictionary with aggregated research findings
        """
        logger.info(f"Starting deep search for: {query}")
        
        # Extract keywords from query if not provided
        if not relevance_keywords:
            # Extract meaningful words from query (simple approach)
            stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'was', 'were', 'from', 'by', 'with', 'as', 'be', 'this', 'that', 'it', 'what', 'how', 'list', 'get', 'find', 'retrieve', 'show', 'all'}
            relevance_keywords = [
                word.lower() for word in re.split(r'\W+', query) 
                if word.lower() not in stop_words and len(word) > 2
            ]
        
        logger.info(f"Using relevance keywords: {relevance_keywords}")
        
        # Track visited URLs to avoid loops
        visited_urls: set = set()
        
        # Aggregated findings
        all_findings: List[Dict[str, Any]] = []
        all_extracted_data: List[Any] = []
        pages_visited = 0
        
        def calculate_relevance_score(text: str) -> float:
            """Calculate how relevant the text is based on keywords."""
            if not text or not relevance_keywords:
                return 0.0
            text_lower = text.lower()
            matches = sum(1 for kw in relevance_keywords if kw in text_lower)
            return matches / len(relevance_keywords) if relevance_keywords else 0.0
        
        def extract_structured_content(html: str, text: str) -> Dict[str, Any]:
            """Try to extract structured data like lists, tables, etc."""
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            structured = {
                "lists": [],
                "tables": [],
                "headings": [],
                "key_paragraphs": []
            }
            
            # Extract headings
            for tag in ['h1', 'h2', 'h3']:
                for heading in soup.find_all(tag):
                    heading_text = heading.get_text(strip=True)
                    if heading_text and len(heading_text) > 3:
                        structured["headings"].append(heading_text)
            
            # Extract lists
            for ul in soup.find_all(['ul', 'ol']):
                items = []
                for li in ul.find_all('li', recursive=False):
                    item_text = li.get_text(strip=True)
                    if item_text and len(item_text) > 2:
                        items.append(item_text)
                if items:
                    structured["lists"].append(items)
            
            # Extract tables
            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = []
                    for cell in tr.find_all(['td', 'th']):
                        cell_text = cell.get_text(strip=True)
                        cells.append(cell_text)
                    if cells:
                        rows.append(cells)
                if rows:
                    structured["tables"].append(rows)
            
            # Extract relevant paragraphs (those containing keywords)
            for p in soup.find_all('p'):
                p_text = p.get_text(strip=True)
                if len(p_text) > 50:  # Skip very short paragraphs
                    score = calculate_relevance_score(p_text)
                    if score > 0.3:  # At least 30% keyword match
                        structured["key_paragraphs"].append(p_text)
            
            return structured
        
        def extract_internal_links(url: str, html: str, be_inclusive: bool = False) -> List[str]:
            """Extract internal links that might be worth following.
            
            Args:
                url: Current page URL
                html: Page HTML content
                be_inclusive: If True, include more links (for first-level exploration)
            """
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse
            
            soup = BeautifulSoup(html, 'html.parser')
            base_domain = urlparse(url).netloc
            internal_links = []
            relevant_links = []  # Prioritized links
            
            for a in soup.find_all('a', href=True):
                href_raw = a.get('href', '')
                # Ensure href is a string
                href = str(href_raw) if href_raw else ''
                if not href:
                    continue
                    
                link_text = a.get_text(strip=True).lower()
                
                # Build absolute URL
                abs_url = urljoin(url, href)
                parsed = urlparse(abs_url)
                
                # Only internal links
                if parsed.netloc != base_domain:
                    continue
                    
                # Skip anchors, javascript, etc.
                if href.startswith('#') or href.startswith('javascript:'):
                    continue
                    
                # Skip media files
                if any(ext in href.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.doc', '.xls']):
                    continue
                
                # Skip already visited
                if abs_url in visited_urls:
                    continue
                
                # Skip duplicates
                if abs_url in internal_links or abs_url in relevant_links:
                    continue
                
                # Prioritize links that seem relevant
                link_relevance = calculate_relevance_score(link_text)
                href_relevance = any(kw in href.lower() for kw in relevance_keywords)
                
                if link_relevance > 0.2 or href_relevance:
                    relevant_links.append(abs_url)
                elif be_inclusive and link_text and len(link_text) > 3:
                    # Include other navigational links for first-level exploration
                    internal_links.append(abs_url)
            
            # Combine: relevant links first, then other internal links
            combined = relevant_links + internal_links
            return combined[:15]  # Return up to 15 links
        
        def visit_and_extract(url: str, depth: int) -> None:
            """Visit a URL and extract relevant information."""
            nonlocal pages_visited
            
            if url in visited_urls:
                return
            if depth > max_depth:
                return
            if pages_visited >= max_results * 3:  # Safety limit
                return
                
            visited_urls.add(url)
            pages_visited += 1
            
            logger.info(f"Deep search visiting ({depth}/{max_depth}): {url}")
            
            try:
                # Try regular scrape first (more reliable), use smart_scrape only if needed
                scrape_result = self.scrape(url=url, extract_text=True, extract_links=True)
                
                if not scrape_result.get("success"):
                    # Try smart_scrape for dynamic content
                    scrape_result = self.smart_scrape(
                        url=url,
                        handle_pagination=False,
                        scroll_pause_time=0.5
                    )
                
                if not scrape_result.get("success"):
                    logger.warning(f"Failed to scrape: {url}")
                    return
                
                # Get text content
                text_content = scrape_result.get("text_content", "") or scrape_result.get("cleaned_text", "")
                # Try to get HTML content - may need to fetch it
                html_content = scrape_result.get("html_content", "") or scrape_result.get("raw_html", "")
                
                # Check if page is in non-English language and try to find English version
                if html_content:
                    english_url = self._detect_and_switch_language(url, html_content)
                    if english_url and english_url not in visited_urls:
                        logger.info(f"🌐 Switching to English version: {english_url}")
                        # Recursively visit the English version instead
                        visit_and_extract(english_url, depth)
                        return  # Don't process non-English version
                
                # If no HTML but we have links from scrape, use them directly
                scraped_links = scrape_result.get("links", [])
                
                title = scrape_result.get("title", "")
                
                # Calculate relevance
                relevance_score = calculate_relevance_score(text_content + " " + title)
                
                # Extract structured data if requested
                structured_data = {}
                if extract_structured_data and html_content:
                    structured_data = extract_structured_content(html_content, text_content)
                
                # Store findings
                finding = {
                    "url": url,
                    "title": title,
                    "depth": depth,
                    "relevance_score": round(relevance_score, 2),
                    "text_preview": text_content[:1000] if text_content else "",
                    "structured_data": structured_data
                }
                all_findings.append(finding)
                
                # Add structured data to aggregated data even with lower relevance (we're already on the site)
                if structured_data:
                    # Always capture lists and tables from target pages
                    if structured_data.get("lists"):
                        all_extracted_data.extend(structured_data["lists"])
                    if structured_data.get("tables"):
                        all_extracted_data.extend(structured_data["tables"])
                
                # Follow internal links if we haven't reached max depth
                if depth < max_depth:
                    # Be more inclusive for first-level exploration (depth 0)
                    be_inclusive = (depth == 0)
                    
                    internal_links = []
                    if html_content:
                        internal_links = extract_internal_links(url, html_content, be_inclusive=be_inclusive)
                    elif scraped_links:
                        # Use links from scrape result when HTML is not available
                        from urllib.parse import urlparse
                        base_domain = urlparse(url).netloc
                        for link_info in scraped_links:
                            if isinstance(link_info, dict):
                                link_url = link_info.get('url', '')
                                link_text = link_info.get('text', '').lower()
                            else:
                                link_url = str(link_info)
                                link_text = ''
                            
                            if not link_url or link_url in visited_urls:
                                continue
                            
                            parsed = urlparse(link_url)
                            # Only internal links
                            if parsed.netloc != base_domain:
                                continue
                            
                            # Skip anchors and special URLs
                            if link_url.endswith('#.') or 'javascript:' in link_url:
                                continue
                            
                            # Check relevance
                            link_relevance = calculate_relevance_score(link_text)
                            if be_inclusive or link_relevance > 0.1 or any(kw in link_url.lower() for kw in relevance_keywords):
                                internal_links.append(link_url)
                        
                        internal_links = internal_links[:15]
                    
                    if internal_links:
                        logger.info(f"Found {len(internal_links)} internal links to explore")
                        for link in internal_links:
                            if len(all_findings) < max_results * 3:  # Safety limit
                                visit_and_extract(link, depth + 1)
                            
            except Exception as e:
                logger.error(f"Error visiting {url}: {str(e)}")
        
        try:
            urls_to_visit = []
            
            if target_url:
                # Start from specific URL
                urls_to_visit = [target_url]
                logger.info(f"Starting deep search from target URL: {target_url}")
            else:
                # Perform search first - with automatic retry on empty results
                search_result = self.search(
                    query=query, 
                    max_results=max_results,
                    retry_on_empty=True,  # Enable retry with query reformulation
                    max_retries=3
                )
                
                if not search_result.get("success"):
                    error_msg = search_result.get('error', 'Unknown error')
                    logger.error(f"Deep search failed - search returned no results: {error_msg}")
                    return {
                        "success": False,
                        "error": f"Search failed: {error_msg}",
                        "query": query,
                        "search_attempts": search_result.get('attempts', [])
                    }
                
                # Extract URLs from search results
                results = search_result.get("results", [])
                urls_to_visit = [r.get("url") or r.get("link") for r in results if r.get("url") or r.get("link")]
                logger.info(f"Found {len(urls_to_visit)} URLs from search")
                
                # Include retry info in logs
                if search_result.get('retry_count', 0) > 0:
                    logger.info(f"Search required {search_result.get('retry_count')} retries to find results")
            
            if not urls_to_visit:
                logger.warning("Deep search found no URLs to explore after all retries")
                return {
                    "success": False,
                    "error": "No URLs found to explore after multiple search attempts",
                    "query": query
                }
            
            # Visit each URL
            for url in urls_to_visit[:max_results]:
                visit_and_extract(url, depth=0)
            
            # Sort findings by relevance
            all_findings.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Flatten and deduplicate extracted data
            flattened_data = []
            seen_items = set()
            for item in all_extracted_data:
                if isinstance(item, list):
                    for subitem in item:
                        if isinstance(subitem, str) and subitem not in seen_items:
                            seen_items.add(subitem)
                            flattened_data.append(subitem)
                        elif isinstance(subitem, list):
                            item_str = str(subitem)
                            if item_str not in seen_items:
                                seen_items.add(item_str)
                                flattened_data.append(subitem)
            
            # Create summary of findings
            top_findings = all_findings[:5]  # Top 5 most relevant pages
            
            summary = {
                "total_pages_visited": pages_visited,
                "relevant_pages_found": len([f for f in all_findings if f.get("relevance_score", 0) > 0.2]),
                "top_sources": [f.get("url") for f in top_findings],
                "extracted_items_count": len(flattened_data)
            }
            
            # Determine if the deep search was actually successful
            # Success criteria: 
            # - Found at least some data, OR
            # - Visited at least one page successfully
            has_extracted_data = len(flattened_data) > 0
            has_findings = len(all_findings) > 0
            is_successful = has_extracted_data or has_findings
            
            if not is_successful:
                logger.warning(f"Deep search completed but found NO useful data after visiting {pages_visited} pages")
                return {
                    "success": False,
                    "error": f"No useful data found after visiting {pages_visited} pages",
                    "query": query,
                    "target_url": target_url,
                    "pages_visited": pages_visited,
                    "summary": summary
                }
            
            logger.info(f"Deep search complete. Visited {pages_visited} pages, found {len(flattened_data)} items")
            
            return {
                "success": True,
                "query": query,
                "target_url": target_url,
                "keywords_used": relevance_keywords,
                "summary": summary,
                "pages_visited": pages_visited,
                "findings": all_findings,
                "extracted_data": flattened_data[:100],  # Limit to 100 items
                "results_count": len(flattened_data),  # Add for consistency with search()
                "search_completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in deep search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "partial_findings": all_findings if all_findings else []
            }

    def research(
        self,
        topic: str,
        target_url: Optional[str] = None,
        search_engines: List[str] = ["duckduckgo"],
        max_sources: int = 5
    ) -> Dict[str, Any]:
        """
        Research a topic like a human researcher would.
        This is a high-level method that combines search and deep extraction.
        
        Args:
            topic: The topic/question to research
            target_url: Optional specific URL to research
            search_engines: List of search engines to use
            max_sources: Maximum number of sources to consult
        
        Returns:
            Comprehensive research findings
        """
        logger.info(f"Starting research on: {topic}")
        
        try:
            # Use deep_search for comprehensive research
            result = self.deep_search(
                query=topic,
                target_url=target_url,
                max_results=max_sources,
                max_depth=2,
                extract_structured_data=True
            )
            
            if not result.get("success"):
                return result
            
            # Create a more research-friendly output
            research_output = {
                "success": True,
                "topic": topic,
                "sources_consulted": result.get("pages_visited", 0),
                "summary": result.get("summary", {}),
                "key_findings": [],
                "extracted_data": result.get("extracted_data", []),
                "source_urls": [],
                "research_completed_at": datetime.now().isoformat()
            }
            
            # Extract key findings from pages
            findings = result.get("findings", [])
            for finding in findings[:10]:  # Top 10 findings
                if finding.get("relevance_score", 0) > 0.1:
                    research_output["source_urls"].append(finding.get("url"))
                    
                    # Add structured data as key findings
                    struct_data = finding.get("structured_data", {})
                    for heading in struct_data.get("headings", []):
                        research_output["key_findings"].append({
                            "type": "heading",
                            "content": heading,
                            "source": finding.get("url")
                        })
                    for para in struct_data.get("key_paragraphs", [])[:3]:
                        research_output["key_findings"].append({
                            "type": "paragraph",
                            "content": para[:500],
                            "source": finding.get("url")
                        })
            
            return research_output
            
        except Exception as e:
            logger.error(f"Error in research: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }

    def execute_task(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AgentExecutionResponse]:
        """
        Execute a web search operation with dual-format support.
        
        Supports three calling conventions:
        1. Legacy positional: execute_task(operation, parameters)
        2. Legacy dict: execute_task({'operation': ..., 'parameters': ...})
        3. Standardized: execute_task(AgentExecutionRequest)
        
        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        
        Returns:
            Legacy dict OR AgentExecutionResponse based on input format
        """
        start_time = time.time()
        return_legacy = True
        operation = None
        parameters = None
        task_dict = None
        
        # Detect calling convention
        # Positional arguments (operation, parameters)
        if len(args) == 2:
            operation, parameters = args
            task_dict = {"operation": operation, "parameters": parameters}
            return_legacy = True
            logger.debug("Legacy positional call")
        
        # Single dict argument
        elif len(args) == 1 and isinstance(args[0], dict):
            task_dict = args[0]
            # Check if standardized request (has task_id and task_description)
            if "task_id" in task_dict and "task_description" in task_dict:
                return_legacy = False
                logger.debug(f"Standardized request call: task_id={task_dict.get('task_id')}")
            else:
                return_legacy = True
                logger.debug("Legacy dict call")
            operation = task_dict.get("operation")
            parameters = task_dict.get("parameters", {})
        
        # Keyword arguments (operation=..., parameters=...)
        elif "operation" in kwargs:
            operation = kwargs.get("operation")
            parameters = kwargs.get("parameters", {})
            task_dict = {"operation": operation, "parameters": parameters}
            return_legacy = True
            logger.debug("Legacy keyword call")
        
        else:
            raise InvalidParameterError(
                parameter_name="task",
                message="Invalid call to execute_task. Use one of:\n"
                "  - execute_task(operation, parameters)\n"
                "  - execute_task({'operation': ..., 'parameters': ...})\n"
                "  - execute_task(AgentExecutionRequest)"
            )
        
        try:
            task_id = task_dict.get("task_id", f"websearch_{int(time.time())}")  # type: ignore
            
            # Ensure parameters is not None
            if parameters is None:
                parameters = {}
            
            # Ensure operation is not None
            if operation is None:
                operation = "unknown"
            
            # Normalize operation aliases to standard operations
            operation_aliases = {
                'web_search': 'search',
                'websearch': 'search',
                'search_web': 'search',
                'web_scrape': 'scrape',
                'webscrape': 'scrape',
                'scrape_web': 'scrape',
                'get': 'fetch',
                'download': 'fetch',
                'retrieve': 'fetch',
                'screenshot': 'capture_screenshot',
                'capture': 'capture_screenshot',
                'paginate': 'handle_pagination',
                'pagination': 'handle_pagination',
                'deep': 'deep_search',
                'browse': 'deep_search',
                'explore': 'deep_search',
                'investigate': 'research',
                'study': 'research'
            }
            
            # Normalize operation to lowercase and check for aliases
            normalized_operation = operation.lower().strip()
            if normalized_operation in operation_aliases:
                actual_operation = operation_aliases[normalized_operation]
                logger.info(f"Executing web search operation: {operation} (normalized to: {actual_operation}, task_id={task_id})")
                operation = actual_operation
            else:
                logger.info(f"Executing web search operation: {operation} (task_id={task_id})")
            
            # Also handle common parameter aliases
            # Map 'num_results' to 'max_results'
            if 'num_results' in parameters and 'max_results' not in parameters:
                parameters['max_results'] = parameters['num_results']
                logger.debug(f"Mapped parameter 'num_results' to 'max_results'")
            
            # Execute the operation using existing methods
            if operation == "search":
                result = self.search(
                    query=parameters.get('query', ''),
                    max_results=parameters.get('max_results', 10),
                    language=parameters.get('language', 'en'),
                    source=parameters.get('source')
                )
            
            elif operation == "scrape":
                result = self.scrape(
                    url=parameters.get('url', ''),
                    extract_text=parameters.get('extract_text', True),
                    extract_links=parameters.get('extract_links', True),
                    extract_images=parameters.get('extract_images', False)
                )
            
            elif operation == "smart_scrape":
                result = self.smart_scrape(
                    url=parameters.get('url', ''),
                    wait_selector=parameters.get('wait_selector'),
                    handle_pagination=parameters.get('handle_pagination', False),
                    max_pages=parameters.get('max_pages', 3),
                    scroll_pause_time=parameters.get('scroll_pause_time', 1.0)
                )
            
            elif operation == "capture_screenshot":
                result = self.capture_screenshot(
                    url=parameters.get('url', ''),
                    selector=parameters.get('selector'),
                    output_dir=parameters.get('output_dir'),
                    wait_selector=parameters.get('wait_selector')
                )
            
            elif operation == "handle_pagination":
                result = self.handle_pagination(
                    url=parameters.get('url', ''),
                    next_button_selector=parameters.get('next_button_selector', ''),
                    content_selector=parameters.get('content_selector', ''),
                    max_pages=parameters.get('max_pages', 5)
                )
            
            elif operation == "fetch":
                result = self.fetch(
                    url=parameters.get('url', ''),
                    save_to_file=parameters.get('save_to_file'),
                    headers=parameters.get('headers')
                )
            
            elif operation == "summarize":
                result = self.summarize(
                    url=parameters.get('url', ''),
                    max_sentences=parameters.get('max_sentences', 5)
                )
            
            elif operation == "deep_search":
                result = self.deep_search(
                    query=parameters.get('query', ''),
                    target_url=parameters.get('target_url') or parameters.get('url'),
                    max_results=parameters.get('max_results', 5),
                    max_depth=parameters.get('max_depth', 2),
                    relevance_keywords=parameters.get('relevance_keywords'),
                    extract_structured_data=parameters.get('extract_structured_data', True)
                )
            
            elif operation == "research":
                result = self.research(
                    topic=parameters.get('topic', parameters.get('query', '')),
                    target_url=parameters.get('target_url') or parameters.get('url'),
                    search_engines=parameters.get('search_engines', ['duckduckgo']),
                    max_sources=parameters.get('max_sources', 5)
                )
            
            else:
                result = {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "supported_operations": self.supported_operations
                }
            
            # Convert legacy result to standardized response
            standard_response = self._convert_to_standard_response(
                result,
                operation,
                task_id,
                start_time
            )
            
            # Publish completion event for event-driven workflows
            self._publish_completion_event(task_id, operation, standard_response)
            
            # Return in requested format
            if return_legacy:
                # Convert back to legacy format for backward compatibility
                return self._convert_to_legacy_response(standard_response)
            else:
                return standard_response
        
        except Exception as e:
            logger.error(f"Error executing WebSearch task: {e}", exc_info=True)
            
            # Create standardized error response
            error = exception_to_error_response(
                e,
                source=self.agent_name,
                task_id=task_dict.get("task_id", "unknown") if task_dict else "unknown"
            )
            
            error_response: AgentExecutionResponse = {
                "status": "failure",
                "success": False,
                "result": {},
                "artifacts": [],
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.agent_name,
                "operation": operation or "unknown",
                "blackboard_entries": [],
                "warnings": []
            }
            # Add error field separately to handle TypedDict
            error_response["error"] = error  # type: ignore
            
            if return_legacy:
                return self._convert_to_legacy_response(error_response)
            else:
                return error_response
    
    def _convert_to_standard_response(
        self,
        legacy_result: Dict[str, Any],
        operation: str,
        task_id: str,
        start_time: float
    ) -> AgentExecutionResponse:
        """Convert legacy result dict to standardized AgentExecutionResponse."""
        success = legacy_result.get("success", False)
        
        # Extract artifacts from result
        artifacts = []
        
        # Handle output files (CSV, JSON, screenshots)
        if "output_file" in legacy_result and success:
            output_file = legacy_result["output_file"]
            if Path(output_file).exists():
                # Determine artifact type from extension
                ext = Path(output_file).suffix.lower()
                artifact_type = ext[1:] if ext else "file"  # Remove the dot
                
                artifacts.append({
                    "type": artifact_type,
                    "path": str(output_file),
                    "size_bytes": Path(output_file).stat().st_size,
                    "description": f"WebSearch {operation} output file"
                })
        
        # Handle screenshot paths
        if "screenshot_path" in legacy_result and success:
            screenshot_path = legacy_result["screenshot_path"]
            if Path(screenshot_path).exists():
                artifacts.append({
                    "type": "png",
                    "path": str(screenshot_path),
                    "size_bytes": Path(screenshot_path).stat().st_size,
                    "description": f"Screenshot from {operation} operation"
                })
        
        # Create blackboard entries for sharing data
        blackboard_entries = []
        if success and "results" in legacy_result:
            blackboard_entries.append({
                "key": f"websearch_results_{task_id}",
                "value": legacy_result.get("results", []),
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        if success and "summary" in legacy_result:
            blackboard_entries.append({
                "key": f"websearch_summary_{task_id}",
                "value": legacy_result.get("summary", ""),
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        # Build standardized response
        response: AgentExecutionResponse = {
            "status": "success" if success else "failure",
            "success": success,
            "result": {
                k: v for k, v in legacy_result.items()
                if k not in ["success", "output_file", "screenshot_path"]
            },
            "artifacts": artifacts,
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "operation": operation,
            "blackboard_entries": blackboard_entries,
            "warnings": []
        }
        
        # Add error field if present (handle TypedDict)
        if not success and "error" in legacy_result:
            response["error"] = create_error_response(  # type: ignore
                error_code="WEBSEARCH_001",
                error_type="execution_error",
                message=legacy_result.get("error", "Unknown error"),
                source=self.agent_name
            )
        
        return response
    
    def _convert_to_legacy_response(self, standard_response: AgentExecutionResponse) -> Dict[str, Any]:
        """Convert standardized response back to legacy format for backward compatibility."""
        legacy = {
            "success": standard_response["success"],
        }
        
        # Add output_file from artifacts
        if standard_response["artifacts"]:
            # Find first CSV, JSON, or other file artifact
            for artifact in standard_response["artifacts"]:
                if artifact["type"] in ["csv", "json", "txt", "md"]:
                    legacy["output_file"] = artifact["path"]
                    break
                elif artifact["type"] == "png":
                    legacy["screenshot_path"] = artifact["path"]
        
        # Add error if present (use .get() for NotRequired field)
        error = standard_response.get("error")  # type: ignore
        if error:
            legacy["error"] = error["message"]  # type: ignore
        
        # Merge result fields into top level (legacy pattern)
        if isinstance(standard_response["result"], dict):
            for key, value in standard_response["result"].items():
                if key not in legacy:
                    legacy[key] = value
        
        return legacy
    
    def _publish_completion_event(
        self,
        task_id: str,
        operation: str,
        response: AgentExecutionResponse
    ):
        """Publish task completion event for event-driven workflows."""
        try:
            # Choose event type based on operation
            if operation in ["search", "deep_search", "research"]:
                event_type = "web_search_completed"
            elif operation in ["scrape", "smart_scrape", "fetch"]:
                event_type = "web_scrape_completed"
            elif "csv" in str(response.get("artifacts", [])):
                event_type = "csv_generated"
            else:
                event_type = "web_operation_completed"
            
            event = create_system_event(
                event_type=event_type,
                event_category="task_lifecycle",
                source_agent=self.agent_name,
                payload={
                    "task_id": task_id,
                    "operation": operation,
                    "success": response["success"],
                    "artifacts": response["artifacts"],
                    "blackboard_keys": [entry["key"] for entry in response["blackboard_entries"]]
                }
            )
            self.event_bus.publish(event)
            logger.debug(f"Published completion event for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to publish completion event: {e}")
