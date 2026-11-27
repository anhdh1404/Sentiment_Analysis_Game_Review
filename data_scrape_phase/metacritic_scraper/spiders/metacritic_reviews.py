"""
Metacritic Game Reviews Spider

Supports the current Nuxt-based Metacritic UI and optional FlareSolverr
for Cloudflare-protected pages and backend API requests.

Usage:
    scrapy crawl metacritic_reviews -a game_name="elden-ring" -a platform="pc"
    scrapy crawl metacritic_reviews -a game_name="elden-ring" -a platform="pc" \
        -a flaresolverr_url="http://localhost:8191"
"""

import html as html_lib
import json
import re
from datetime import datetime
from urllib.parse import urlparse

import scrapy
from bs4 import BeautifulSoup

from metacritic_scraper.items import GameReviewItem


class NuxtPayload:
    """Decode Nuxt's devalue-serialized __NUXT_DATA__ array."""

    SPECIAL = {
        -1: None,
        -2: None,
        -3: float("nan"),
        -4: float("inf"),
        -5: float("-inf"),
        -6: -0.0,
    }

    def __init__(self, payload):
        self.payload = payload
        self.memo = {}

    def resolve(self, ref):
        if isinstance(ref, int):
            if ref < 0:
                return self.SPECIAL.get(ref)
            if ref in self.memo:
                return self.memo[ref]
            value = self.payload[ref]
            if isinstance(value, dict):
                result = {}
                self.memo[ref] = result
                for key, child in value.items():
                    result[key] = self.resolve(child)
                return result
            if isinstance(value, list):
                result = []
                self.memo[ref] = result
                for child in value:
                    result.append(self.resolve(child))
                return result
            self.memo[ref] = value
            return value
        return ref


class MetacriticReviewsSpider(scrapy.Spider):
    name = "metacritic_reviews"
    allowed_domains = ["metacritic.com", "localhost", "127.0.0.1"]

    custom_settings = {
        "DOWNLOAD_DELAY": 2,
        "CONCURRENT_REQUESTS": 2,
        "AUTOTHROTTLE_ENABLED": True,
        "DOWNLOAD_TIMEOUT": 90,
        "RETRY_TIMES": 3,
    }

    def __init__(
        self,
        game_url=None,
        game_name=None,
        platform=None,
        max_reviews_per_platform=None,
        max_reviews=None,
        flaresolverr_url=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        limit = max_reviews_per_platform or max_reviews
        self.max_reviews_per_platform = int(limit) if limit else None
        self.flaresolverr_url = (flaresolverr_url or "").rstrip("/") or None
        if self.flaresolverr_url:
            host = urlparse(self.flaresolverr_url).hostname
            if host and host not in self.allowed_domains:
                self.allowed_domains = [*self.allowed_domains, host]
        self.total_reviews_count = 0
        self.platform_reviews_count = {}
        self.seen_review_ids = set()
        self.game_info = {
            "game_title": None,
            "game_platform": platform,
            "game_url": None,
            "game_metascore": None,
            "game_release_date": None,
            "game_genre": None,
            "game_developer": None,
            "game_publisher": None,
        }

        slug, url_platform = self._slug_from_url(game_url)
        self.game_slug = game_name or slug
        self.platform = platform or url_platform

        if not self.game_slug:
            raise ValueError("Either game_url or game_name is required")

        self.review_url = self._build_review_url(self.game_slug, self.platform)
        self.game_info["game_url"] = f"https://www.metacritic.com/game/{self.game_slug}/"

    @staticmethod
    def _slug_from_url(game_url):
        if not game_url:
            return None, None
        path = urlparse(game_url).path.strip("/")
        parts = [part for part in path.split("/") if part]
        if not parts or parts[0] != "game":
            return None, None
        review_suffixes = {"user-reviews", "critic-reviews"}
        if len(parts) >= 3 and parts[-1] in review_suffixes:
            if len(parts) == 3:
                return parts[1], None
            return parts[2], parts[1]
        if len(parts) >= 3:
            return parts[2], parts[1]
        if len(parts) == 2:
            return parts[1], None
        return None, None

    @staticmethod
    def _build_review_url(slug, platform=None):
        url = f"https://www.metacritic.com/game/{slug}/user-reviews/"
        if platform and platform not in ("all", "multiplatform", "unknown"):
            url += f"?platform={platform}"
        return url

    async def start(self):
        for request in self.start_requests():
            yield request

    def start_requests(self):
        self.logger.info("Starting Metacritic reviews for %s", self.review_url)
        if self.flaresolverr_url:
            yield self._flaresolverr_request(
                self.review_url,
                self.parse_flaresolverr_page,
                meta={"target_url": self.review_url},
            )
            return

        yield scrapy.Request(
            url=self.review_url,
            callback=self.parse_playwright_page,
            errback=self.handle_error,
            meta={
                "playwright": True,
                "playwright_page_goto_kwargs": {
                    "wait_until": "domcontentloaded",
                    "timeout": 90000,
                },
                "playwright_page_methods": [
                    {"wait_for_timeout": 5000},
                ],
            },
        )

    def _flaresolverr_request(self, target_url, callback, meta=None):
        request_meta = dict(meta or {})
        request_meta["target_url"] = target_url
        return scrapy.Request(
            url=f"{self.flaresolverr_url}/v1",
            method="POST",
            body=json.dumps(
                {
                    "cmd": "request.get",
                    "url": target_url,
                    "maxTimeout": 90000,
                }
            ),
            headers={"Content-Type": "application/json"},
            callback=callback,
            errback=self.handle_error,
            meta=request_meta,
            dont_filter=True,
        )

    def _solution_from_response(self, response):
        try:
            payload = json.loads(response.text)
        except (TypeError, ValueError) as exc:
            self.logger.error("Invalid FlareSolverr response: %s", exc)
            return None, None

        if payload.get("status") != "ok":
            self.logger.error(
                "FlareSolverr failed for %s: %s",
                response.meta.get("target_url"),
                payload.get("message"),
            )
            return None, None

        solution = payload.get("solution") or {}
        return solution.get("response") or "", solution.get("url")

    def _decode_nuxt(self, html):
        soup = BeautifulSoup(html, "html.parser")
        script = soup.find("script", id="__NUXT_DATA__")
        if not script or not script.string:
            return None

        try:
            raw = json.loads(script.string)
        except json.JSONDecodeError:
            return None

        nuxt = NuxtPayload(raw)
        for index, value in enumerate(raw):
            if not isinstance(value, dict) or "components" not in value:
                continue
            decoded = nuxt.resolve(index)
            if isinstance(decoded, dict) and decoded.get("components"):
                return decoded
        return None

    @staticmethod
    def _find_component(nuxt, component_name):
        for component in nuxt.get("components", []):
            meta = component.get("meta") or {}
            if meta.get("componentName") == component_name:
                return component
        return None

    def _update_game_info(self, nuxt, page_url):
        product = self._find_component(nuxt, "product")
        if not product:
            return

        item = (product.get("data") or {}).get("item") or {}
        if not isinstance(item, dict):
            return

        critic_score = item.get("criticScoreSummary") or {}
        companies = ((item.get("production") or {}).get("companies") or [])
        developer = next(
            (company.get("name") for company in companies if company.get("typeName") == "Developer"),
            None,
        )
        publisher = next(
            (company.get("name") for company in companies if company.get("typeName") == "Publisher"),
            None,
        )
        genres = [genre.get("name") for genre in (item.get("genres") or []) if genre.get("name")]

        self.game_info.update(
            {
                "game_title": item.get("title"),
                "game_platform": item.get("platform"),
                "game_url": self._absolute_url(item.get("url")) or self.game_info.get("game_url"),
                "game_metascore": critic_score.get("score"),
                "game_release_date": item.get("releaseDateText") or item.get("releaseDate"),
                "game_genre": ", ".join(genres) or None,
                "game_developer": developer,
                "game_publisher": publisher,
            }
        )

        reviews = self._find_component(nuxt, "user-reviews")
        if reviews:
            review_items = ((reviews.get("data") or {}).get("items") or [])
            if review_items:
                product_info = review_items[0].get("reviewedProduct") or {}
                if product_info.get("title"):
                    self.game_info["game_title"] = product_info["title"]

    @staticmethod
    def _absolute_url(url):
        if not url:
            return None
        if url.startswith("http"):
            return url
        return f"https://www.metacritic.com{url}"

    def parse_flaresolverr_page(self, response):
        html, final_url = self._solution_from_response(response)
        if not html:
            return

        final_url = final_url or response.meta.get("target_url") or self.review_url
        nuxt = self._decode_nuxt(html)
        if nuxt:
            self._update_game_info(nuxt, final_url)
            reviews = self._find_component(nuxt, "user-reviews")
            if reviews:
                review_data = reviews.get("data") or {}
                yield from self._yield_reviews(review_data.get("items") or [], final_url)
                next_url = ((reviews.get("links") or {}).get("next") or {}).get("href")
                if next_url and self._should_continue():
                    yield self._flaresolverr_request(
                        next_url,
                        self.parse_flaresolverr_api,
                        meta={"target_url": next_url},
                    )
                return

        self.logger.warning("Nuxt payload not found; falling back to DOM parsing")
        yield from self._parse_dom_reviews(html, final_url)

    def parse_flaresolverr_api(self, response):
        html, final_url = self._solution_from_response(response)
        if not html:
            return

        try:
            data = self._parse_json_body(html)
        except ValueError as exc:
            self.logger.error("Could not parse FlareSolverr API response: %s", exc)
            return

        review_data = data.get("data") or {}
        yield from self._yield_reviews(review_data.get("items") or [], self.review_url)

        next_url = ((data.get("links") or {}).get("next") or {}).get("href")
        if next_url and self._should_continue():
            yield self._flaresolverr_request(
                next_url,
                self.parse_flaresolverr_api,
                meta={"target_url": next_url},
            )

    def parse_playwright_page(self, response):
        html = response.text
        nuxt = self._decode_nuxt(html)
        if nuxt:
            self._update_game_info(nuxt, response.url)
            reviews = self._find_component(nuxt, "user-reviews")
            if reviews:
                review_data = reviews.get("data") or {}
                yield from self._yield_reviews(review_data.get("items") or [], response.url)
                next_url = ((reviews.get("links") or {}).get("next") or {}).get("href")
                if next_url:
                    self.logger.warning(
                        "More reviews are available at %s. Use FlareSolverr to paginate.",
                        next_url,
                    )
                return

        self.logger.warning(
            "Nuxt payload not found. Metacritic may be blocking Playwright with Cloudflare; "
            "run with -a flaresolverr_url=http://localhost:8191"
        )
        yield from self._parse_dom_reviews(html, response.url)

    @staticmethod
    def _parse_json_body(body):
        text = body.strip()
        if text.startswith("{") or text.startswith("["):
            return json.loads(text)

        match = re.search(r"<pre[^>]*>(.*?)</pre>", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = html_lib.unescape(match.group(1))
        else:
            soup = BeautifulSoup(text, "html.parser")
            pre = soup.find("pre")
            text = pre.get_text() if pre else soup.get_text()

        return json.loads(text)

    def _should_continue(self):
        if not self.max_reviews_per_platform:
            return True
        return self.total_reviews_count < self.max_reviews_per_platform

    def _yield_reviews(self, reviews, page_url):
        for review in reviews:
            if not self._should_continue():
                return
            item = self._review_to_item(review, page_url)
            if item:
                yield item

    def _review_to_item(self, review, page_url):
        if not isinstance(review, dict):
            return None

        review_id = review.get("id")
        if not review_id:
            review_id = (
                f"{self.game_info.get('game_title', 'unknown')}_"
                f"{review.get('author') or 'anonymous'}_"
                f"{review.get('date') or 'no-date'}"
            )
            review_id = re.sub(r"[^\w\-]", "_", review_id)

        if review_id in self.seen_review_ids:
            return None
        self.seen_review_ids.add(review_id)

        review_text = review.get("quote") or review.get("review") or ""
        if isinstance(review_text, str):
            review_text = html_lib.unescape(review_text)
            review_text = " ".join(review_text.split())
        if not review_text:
            return None

        score = review.get("score")
        try:
            score = int(float(score) * 10)
        except (TypeError, ValueError):
            score = None

        category = None
        if score is not None:
            if score >= 75:
                category = "positive"
            elif score >= 50:
                category = "mixed"
            else:
                category = "negative"

        platform = review.get("platform") or self.game_info.get("game_platform")
        reviewed_product = review.get("reviewedProduct") or {}

        item = GameReviewItem()
        item["game_title"] = self.game_info.get("game_title") or reviewed_product.get("title")
        item["game_platform"] = platform
        item["game_url"] = self.game_info.get("game_url")
        item["game_metascore"] = self.game_info.get("game_metascore")
        item["game_release_date"] = self.game_info.get("game_release_date")
        item["game_genre"] = self.game_info.get("game_genre")
        item["game_developer"] = self.game_info.get("game_developer")
        item["game_publisher"] = self.game_info.get("game_publisher")

        item["review_id"] = review_id
        item["review_text"] = review_text
        item["review_score"] = score
        item["review_category"] = category
        item["reviewer_name"] = review.get("author")
        item["review_date"] = review.get("date")
        item["review_url"] = page_url
        item["scraped_at"] = datetime.now().isoformat()

        self.total_reviews_count += 1
        self.platform_reviews_count[platform] = (
            self.platform_reviews_count.get(platform, 0) + 1
        )
        return item

    def _parse_dom_reviews(self, html, page_url):
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select('[data-testid="review-card"]')
        if not cards:
            self.logger.warning("No review cards found on %s", page_url)
            return

        title = soup.select_one("h1")
        if title and not self.game_info.get("game_title"):
            self.game_info["game_title"] = title.get_text(" ", strip=True)

        for card in cards:
            if not self._should_continue():
                return

            quote = card.select_one('[data-testid="review-quote-text"]')
            header = card.select_one('[data-testid="review-card-header"]')
            date = card.select_one('[data-testid="review-card-date"]')
            platform = card.select_one('[data-testid="review-platform"]')
            score_el = card.select_one(".c-siteReviewScore")

            score = None
            if score_el:
                score_match = re.search(r"\d+(?:\.\d+)?", score_el.get_text(" ", strip=True))
                if score_match:
                    score = int(float(score_match.group()) * 10)

            author = None
            if header:
                author_text = header.get_text(" ", strip=True)
                author_text = re.sub(r"^User score \d+ out of \d+\s*", "", author_text)
                author = author_text.strip() or None

            review = {
                "id": None,
                "quote": quote.get_text(" ", strip=True) if quote else None,
                "score": score / 10 if score is not None else None,
                "author": author,
                "date": date.get_text(" ", strip=True) if date else None,
                "platform": platform.get_text(" ", strip=True) if platform else None,
            }
            item = self._review_to_item(review, page_url)
            if item:
                yield item

    def handle_error(self, failure):
        self.logger.error("Request failed: %s", failure.request.url)
        self.logger.error("Error: %s", failure.value)
        if "TimeoutError" in str(failure.type) or "TCPTimedOutError" in str(failure.type):
            self.logger.warning(
                "Timeout. Check network, increase DOWNLOAD_DELAY, or use FlareSolverr."
            )
