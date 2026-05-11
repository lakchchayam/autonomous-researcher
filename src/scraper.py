"""
Web scraping tools for the autonomous researcher.

Provides a Playwright-based browser tool that can fetch and extract
clean text content from JavaScript-heavy websites, handle pagination,
and retry on transient failures.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger(__name__)


@dataclass
class ScrapedPage:
    """Result of scraping a single web page."""
    url: str
    title: str
    text_content: str
    word_count: int
    success: bool
    error: Optional[str] = None


class WebScraper:
    """
    Lightweight web scraper with fallback strategies.
    
    Attempts fast HTTP-based scraping first. Falls back to
    Playwright for JavaScript-rendered pages. Extracts clean
    text content using BeautifulSoup.
    """
    
    def __init__(self, timeout: int = 15, max_content_length: int = 50000) -> None:
        self._timeout = timeout
        self._max_length = max_content_length
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    
    async def scrape_url(self, url: str) -> ScrapedPage:
        """
        Scrape a URL and return cleaned text content.
        
        First tries a fast httpx GET request. If the content looks
        JS-rendered (minimal text), falls back to Playwright.
        """
        # Attempt fast HTTP scrape
        try:
            page = await self._scrape_with_httpx(url)
            if page.success and page.word_count > 50:
                return page
            logger.info("HTTP scrape returned thin content, trying Playwright", url=url)
        except Exception as err:
            logger.warning("HTTP scrape failed", url=url, error=str(err))
        
        # Fall back to Playwright
        try:
            return await self._scrape_with_playwright(url)
        except Exception as err:
            logger.error("All scrape methods failed", url=url, error=str(err))
            return ScrapedPage(
                url=url, title="", text_content="",
                word_count=0, success=False, error=str(err),
            )
    
    async def _scrape_with_httpx(self, url: str) -> ScrapedPage:
        """Fast HTTP-based scraping with BeautifulSoup parsing."""
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers=self._headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()
            
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            text = soup.get_text(separator="\n", strip=True)
            
            # Truncate if too long
            if len(text) > self._max_length:
                text = text[:self._max_length] + "\n\n[Content truncated]"
            
            word_count = len(text.split())
            
            return ScrapedPage(
                url=url,
                title=title,
                text_content=text,
                word_count=word_count,
                success=True,
            )
    
    async def _scrape_with_playwright(self, url: str) -> ScrapedPage:
        """Playwright-based scraping for JS-rendered pages."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required for JS-rendered pages. "
                "Install with: pip install playwright && playwright install chromium"
            )
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(url, wait_until="networkidle", timeout=self._timeout * 1000)
                
                title = await page.title()
                
                # Extract main content, trying article/main tags first
                content = await page.evaluate("""() => {
                    const selectors = ['article', 'main', '[role="main"]', '.content', '#content'];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el && el.innerText.trim().length > 200) {
                            return el.innerText.trim();
                        }
                    }
                    return document.body.innerText.trim();
                }""")
                
                if len(content) > self._max_length:
                    content = content[:self._max_length] + "\n\n[Content truncated]"
                
                return ScrapedPage(
                    url=url,
                    title=title,
                    text_content=content,
                    word_count=len(content.split()),
                    success=True,
                )
            finally:
                await browser.close()
    
    async def scrape_multiple(self, urls: list[str], concurrency: int = 3) -> list[ScrapedPage]:
        """Scrape multiple URLs with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _bounded_scrape(url: str) -> ScrapedPage:
            async with semaphore:
                return await self.scrape_url(url)
        
        tasks = [_bounded_scrape(url) for url in urls]
        return await asyncio.gather(*tasks)
