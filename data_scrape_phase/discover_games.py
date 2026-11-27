"""
Discover Games Script for Metacritic Scraper

This script browses Metacritic's game listings to discover all available games,
their platforms, and URLs. It saves this information to a file that can then be
used by the scraping script.

Usage:
    python discover_games.py --max-pages 10 --platform all
    python discover_games.py --platform playstation-5 --min-score 70
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import shutil
import subprocess
import time
import json
import csv
import argparse
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import random


DEFAULT_FLARESOLVERR_URL = "http://localhost:8191"
FLARESOLVERR_CONTAINER = "flaresolverr"


def resolve_output_path(filename):
    """Resolve output path without duplicating the data/ prefix."""
    path = Path(filename)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return path
    return Path('data') / path


class FlareSolverrResponse:
    """Minimal requests.Response-compatible wrapper for FlareSolverr."""

    def __init__(self, status_code, content, url):
        self.status_code = status_code
        self.content = content.encode('utf-8') if isinstance(content, str) else content
        self.text = self.content.decode('utf-8', errors='replace')
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error for url: {self.url}")


class NuxtPayload:
    """Decode Nuxt's devalue-serialized __NUXT_DATA__ array."""

    SPECIAL = {
        -1: None,
        -2: None,
        -3: float('nan'),
        -4: float('inf'),
        -5: float('-inf'),
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


class GameDiscoverer:
    """Discovers games from Metacritic browse pages"""
    
    def __init__(self, delay=1, max_pages=None, incremental=True, split_files=1, flaresolverr_url=None,
                 auto_start_flaresolverr=True):
        self.base_url = "https://www.metacritic.com"
        self.delay = delay
        self.max_pages = max_pages
        self.incremental = incremental
        self.split_files = max(1, split_files)  # At least 1 file
        self.flaresolverr_url = flaresolverr_url.rstrip('/') if flaresolverr_url else None
        self.auto_start_flaresolverr = auto_start_flaresolverr
        self._flaresolverr_start_attempted = False
        self.discovered_games = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.output_file = None
        self.output_files = []  # List of output file paths when splitting
        self.discovered_count = 0
        self.current_file_index = 0
    
    def _initialize_output_file(self, output_format='txt', filename=None):
        """Initialize the output file(s) for incremental saving"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"discovered_games_{timestamp}.{output_format}"
        
        # If splitting into multiple files
        if self.split_files > 1:
            self.output_files = []
            base_path = resolve_output_path(filename)
            base_name = base_path.stem
            extension = base_path.suffix.lstrip('.') or output_format
            
            for i in range(self.split_files):
                file_path = base_path.parent / f"{base_name}_part{i+1}.{extension}"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_files.append(file_path)
                
                # Initialize each file
                if output_format == 'json':
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('[\n')
                elif output_format == 'csv':
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['title', 'game_name_slug', 'platform_slug', 'url', 'metascore', 'release_date', 'discovered_at'])
            
            self.output_file = self.output_files[0]  # Set first file as current
            return self.output_files
        else:
            # Single file mode
            self.output_file = resolve_output_path(filename)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # For JSON, start with an array
            if output_format == 'json':
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write('[\n')
            
            # For CSV, write headers
            elif output_format == 'csv':
                with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['title', 'game_name_slug', 'url', 'metascore', 'release_date', 'discovered_at'])
            
            return self.output_file
    
    def _append_game_to_file(self, game_data, output_format='txt'):
        """Append a single game to the output file immediately"""
        if not self.output_file:
            return
        
        try:
            # Determine which file to write to when splitting
            if self.split_files > 1:
                # Distribute games evenly across files
                file_index = self.discovered_count % self.split_files
                current_file = self.output_files[file_index]
                
                # Track items per file for JSON comma handling
                items_in_file = self.discovered_count // self.split_files
            else:
                current_file = self.output_file
                items_in_file = self.discovered_count
            
            if output_format == 'txt':
                # Format: game_slug|platform_slug
                platform_slug = game_data.get('platform_slug') or 'multiplatform'
                with open(current_file, 'a', encoding='utf-8') as f:
                    f.write(f"{game_data['game_name_slug']}|{platform_slug}\n")
            
            elif output_format == 'json':
                with open(current_file, 'a', encoding='utf-8') as f:
                    # Add comma if this file already has entries
                    if items_in_file > 0:
                        f.write(',\n')
                    
                    # Write the game data (closing bracket is added in finalize)
                    json_line = json.dumps(game_data, indent=2, ensure_ascii=False)
                    json_line = '\n'.join('  ' + line for line in json_line.split('\n'))
                    f.write(json_line)
            
            elif output_format == 'csv':
                with open(current_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        game_data.get('title', ''),
                        game_data.get('game_name_slug', ''),
                        game_data.get('platform_slug', ''),
                        game_data.get('url', ''),
                        game_data.get('metascore', ''),
                        game_data.get('release_date', ''),
                        game_data.get('discovered_at', '')
                    ])
            
            self.discovered_count += 1
            
        except Exception as e:
            print(f"    ⚠ Error saving game to file: {e}")
    
    def _finalize_output_file(self, output_format='txt'):
        """Finalize the output file(s) (close JSON array, etc.)"""
        if not self.output_file:
            return
        
        try:
            if output_format == 'json':
                # If splitting, finalize all files
                if self.split_files > 1:
                    for file_path in self.output_files:
                        with open(file_path, 'a', encoding='utf-8') as f:
                            f.write(']\n')
                else:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        f.write(']\n')
        except Exception as e:
            print(f"⚠ Error finalizing output file: {e}")
    
    def _flaresolverr_is_healthy(self, url):
        try:
            response = requests.get(f"{url.rstrip('/')}/health", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def _start_flaresolverr_container(self):
        if self._flaresolverr_start_attempted or not self.auto_start_flaresolverr:
            return False
        self._flaresolverr_start_attempted = True

        if not shutil.which("docker"):
            return False

        try:
            check = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name=^{FLARESOLVERR_CONTAINER}$", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if FLARESOLVERR_CONTAINER not in check.stdout.split():
                return False

            print(f"  ⏳ FlareSolverr chưa chạy, đang start container '{FLARESOLVERR_CONTAINER}'...")
            started = subprocess.run(
                ["docker", "start", FLARESOLVERR_CONTAINER],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if started.returncode != 0:
                print(f"  ⚠ Không start được FlareSolverr: {started.stderr.strip()}")
                return False

            for _ in range(20):
                time.sleep(1)
                if self._flaresolverr_is_healthy(DEFAULT_FLARESOLVERR_URL):
                    print("  ✓ FlareSolverr đã sẵn sàng")
                    return True
        except Exception as exc:
            print(f"  ⚠ Không thể tự start FlareSolverr: {exc}")

        return False

    def _resolve_flaresolverr_url(self):
        candidates = []
        for candidate in (
            self.flaresolverr_url,
            os.getenv('FLARESOLVERR_URL'),
            DEFAULT_FLARESOLVERR_URL,
        ):
            if not candidate:
                continue
            normalized = candidate.rstrip('/')
            if normalized not in candidates:
                candidates.append(normalized)

        for candidate in candidates:
            if self._flaresolverr_is_healthy(candidate):
                if candidate != self.flaresolverr_url:
                    print(f"  ✓ Tự động dùng FlareSolverr: {candidate}")
                self.flaresolverr_url = candidate
                return candidate

        if self._start_flaresolverr_container():
            for candidate in candidates:
                if self._flaresolverr_is_healthy(candidate):
                    self.flaresolverr_url = candidate
                    return candidate

        return None

    def _fetch_via_flaresolverr(self, url):
        response = requests.post(
            f"{self.flaresolverr_url}/v1",
            json={
                "cmd": "request.get",
                "url": url,
                "maxTimeout": 90000,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            raise RuntimeError(payload.get("message", "FlareSolverr request failed"))
        solution = payload.get("solution") or {}
        return FlareSolverrResponse(
            int(solution.get("status", 200)),
            solution.get("response", ""),
            solution.get("url", url),
        )

    def _fetch(self, url):
        """Fetch a URL, optionally through FlareSolverr to bypass Cloudflare."""
        if not self.flaresolverr_url:
            self._resolve_flaresolverr_url()

        if self.flaresolverr_url:
            try:
                return self._fetch_via_flaresolverr(url)
            except Exception as exc:
                print(f"  ⚠ FlareSolverr failed: {exc}. Falling back to requests...")
                self.flaresolverr_url = None

        response = self.session.get(url, timeout=30)
        if response.status_code in (403, 429):
            if self._resolve_flaresolverr_url():
                print("  ↻ Request trực tiếp bị chặn, thử lại qua FlareSolverr...")
                return self._fetch_via_flaresolverr(url)

        return response

    def discover_from_browse(self, platform=None, min_score=None, sort='score', output_format='txt', filename=None):
        """
        Discover games from Metacritic browse pages
        
        Args:
            platform: Platform filter (e.g., 'playstation-5', 'pc', 'all')
            min_score: Minimum metascore filter (0-100)
            sort: Sort order ('score', 'date', 'name')
            output_format: Output file format ('txt', 'json', 'csv')
            filename: Custom filename (optional)
        """
        # Initialize output file for incremental saving
        if self.incremental:
            output_file = self._initialize_output_file(output_format, filename)
            if self.split_files > 1:
                print(f"Incremental output (split into {self.split_files} files):")
                for i, f in enumerate(self.output_files, 1):
                    print(f"  Part {i}: {f}")
            else:
                print(f"Incremental output: {output_file}")
        
        print("=" * 70)
        print("GAME DISCOVERY - Metacritic Browse")
        print("=" * 70)
        print(f"Platform: {platform or 'all'}")
        print(f"Min Score: {min_score or 'none'}")
        print(f"Max Pages: {self.max_pages or 'unlimited'}")
        print(f"Incremental save: {'Yes' if self.incremental else 'No'}")
        print(f"Split into files: {self.split_files}")
        print("=" * 70)
        print()
        
        # Build browse URL
        # Example: https://www.metacritic.com/browse/game/
        browse_url = f"{self.base_url}/browse/game/"
        
        page = 1
        games_found = 0
        
        while True:
            if self.max_pages and self.max_pages > 0 and page > self.max_pages:
                print(f"\n✓ Reached max pages limit ({self.max_pages})")
                break
            
            # Metacritic uses pagination like: /browse/game/?page=1
            url = f"{browse_url}?page={page}"
            
            print(f"\n[Page {page}] Fetching: {url}")
            
            try:
                response = self._fetch(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # New Metacritic pages embed the game list in __NUXT_DATA__.
                page_games_data, total_available = self._extract_games_from_nuxt(soup)
                
                # Fallback for older layouts.
                if not page_games_data:
                    game_cards = (
                        soup.select('div.c-finderProductCard') or
                        soup.select('td.clamp-summary-wrap') or
                        soup.select('div.product_wrap') or
                        []
                    )
                    page_games_data = [
                        self._extract_game_from_card(card, soup)
                        for card in game_cards
                    ]
                    page_games_data = [game for game in page_games_data if game]
                
                if not page_games_data:
                    print(f"  ⚠ No games found on page {page}")
                    break
                
                print(f"  Found {len(page_games_data)} games")
                if total_available is not None:
                    print(f"  Total available: {total_available}")
                
                page_games = 0
                for game_data in page_games_data:
                    # Apply filters
                    if min_score and game_data.get('metascore'):
                        try:
                            if int(game_data['metascore']) < min_score:
                                continue
                        except (ValueError, TypeError):
                            pass
                    
                    if platform and platform != 'all':
                        platform_slugs = game_data.get('platform_slugs') or []
                        if platform not in platform_slugs:
                            continue
                    
                    self.discovered_games.append(game_data)
                    page_games += 1
                    games_found += 1
                    
                    # Save incrementally if enabled
                    if self.incremental:
                        self._append_game_to_file(game_data, output_format)
                    
                    print(f"  [{games_found}] {game_data['title']} - Score: {game_data.get('metascore', 'N/A')}")
                
                if page_games == 0 and page > 1:
                    print(f"  No games matched filters on page {page}")
                
                # Stop when we've reached the total reported by the payload.
                if total_available is not None and page * len(page_games_data) >= total_available:
                    print(f"\n✓ No more pages (last page: {page})")
                    break
                
                page += 1
                
                # Respectful delay
                delay_time = self.delay + random.uniform(0, 1)
                print(f"  Waiting {delay_time:.1f}s before next request...")
                time.sleep(delay_time)
                
            except requests.RequestException as e:
                print(f"  ✗ Error fetching page {page}: {e}")
                if '403' in str(e):
                    if self.flaresolverr_url:
                        print("    FlareSolverr đang chạy nhưng vẫn bị Cloudflare chặn.")
                        print("    Thử tăng --delay hoặc kiểm tra container FlareSolverr.")
                    else:
                        print("    Không vượt được Cloudflare và chưa có FlareSolverr healthy.")
                        print("    Chạy:")
                        print("    docker start flaresolverr")
                        print("    python data_scrape_phase/discover_games.py ... --flaresolverr http://localhost:8191")
                break
            except Exception as e:
                print(f"  ✗ Error processing page {page}: {e}")
                break
        
        # Finalize output file if incremental
        if self.incremental:
            self._finalize_output_file(output_format)
        
        print("\n" + "=" * 70)
        print(f"DISCOVERY COMPLETE - Found {len(self.discovered_games)} games")
        if self.incremental:
            if self.split_files > 1:
                print(f"Saved to {self.split_files} files:")
                for i, f in enumerate(self.output_files, 1):
                    # Count games in each file
                    games_in_file = len([g for idx, g in enumerate(self.discovered_games) if idx % self.split_files == (i-1)])
                    print(f"  Part {i}: {f} ({games_in_file} games)")
            elif self.output_file:
                print(f"Saved to: {self.output_file}")
        print("=" * 70)
        
        return self.discovered_games
    
    @staticmethod
    def _slugify_platform(name):
        if not name:
            return 'multiplatform'
        slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
        aliases = {
            'switch': 'nintendo-switch',
            'switch-2': 'nintendo-switch-2',
            'ps5': 'playstation-5',
            'ps4': 'playstation-4',
            'ps3': 'playstation-3',
            'ps2': 'playstation-2',
            'ps1': 'playstation',
        }
        return aliases.get(slug, slug)

    def _extract_games_from_nuxt(self, soup):
        """Extract the game list from Metacritic's Nuxt payload.

        Returns (games, total_available). total_available is None if unknown.
        """
        script = soup.find('script', id='__NUXT_DATA__')
        if not script or not script.string:
            return [], None

        try:
            raw = json.loads(script.string)
        except json.JSONDecodeError:
            return [], None

        nuxt = NuxtPayload(raw)
        for index, value in enumerate(raw):
            if not isinstance(value, dict) or 'items' not in value or 'total' not in value:
                continue

            decoded = nuxt.resolve(index)
            items = decoded.get('items')
            if not isinstance(items, list) or not items:
                continue
            if not isinstance(items[0], dict) or items[0].get('type') != 'game-title':
                continue

            games = [self._game_from_nuxt_item(item) for item in items]
            games = [game for game in games if game]
            return games, decoded.get('total')

        return [], None

    def _game_from_nuxt_item(self, item):
        slug = item.get('slug')
        if not slug:
            return None

        platforms = [
            platform.get('name')
            for platform in (item.get('platforms') or [])
            if isinstance(platform, dict) and platform.get('name')
        ]
        platform_slugs = [self._slugify_platform(name) for name in platforms]
        critic_score = item.get('criticScoreSummary') or {}
        user_score = item.get('userScore') or {}

        return {
            'title': item.get('title'),
            'game_name_slug': slug,
            'url': f"{self.base_url}/game/{slug}/",
            'metascore': critic_score.get('score'),
            'user_score': user_score.get('score'),
            'release_date': item.get('releaseDate'),
            'rank': item.get('rank'),
            'platform': platforms[0] if platforms else None,
            'platform_slug': platform_slugs[0] if platform_slugs else 'multiplatform',
            'platforms': platforms,
            'platform_slugs': platform_slugs,
            'discovered_at': datetime.now().isoformat(),
        }

    def _extract_game_from_card(self, card, soup):
        """Extract game information from a card element"""
        try:
            # Try to extract title
            title_elem = (
                card.select_one('h3.c-finderProductCard_titleHeading') or
                card.select_one('a.title h3') or
                card.select_one('h3')
            )
            
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            
            # Try to extract URL
            link_elem = (
                card.select_one('a.c-finderProductCard_container') or
                card.select_one('a.title') or
                card.select_one('a[href*="/game/"]')
            )
            
            if not link_elem:
                return None
            
            game_url = urljoin(self.base_url, link_elem.get('href', ''))
            
            # Extract game name from URL
            # URL format: /game/{game-name}/ (new Metacritic format - no platform in browse URL)
            url_parts = [p for p in urlparse(game_url).path.split('/') if p]  # Filter out empty strings
            
            # Expected: ['game', 'game-name']
            if len(url_parts) >= 2 and url_parts[0] == 'game':
                game_name_slug = url_parts[1]
            else:
                print(f"    ⚠ Cannot parse URL: {game_url}")
                return None
            
            # Try to extract metascore
            score_elem = (
                card.select_one('div.c-siteReviewScore span') or
                card.select_one('div.c-finderProductCard_score span') or
                card.select_one('div.metascore_w') or
                card.select_one('.c-siteReviewScore')
            )
            metascore = score_elem.get_text(strip=True) if score_elem else None
            
            # Try to extract release date
            date_elem = (
                card.select_one('div.c-finderProductCard_meta span') or
                card.select_one('span.release_date') or
                card.select_one('.c-finderProductCard_releaseDate')
            )
            release_date = date_elem.get_text(strip=True) if date_elem else None
            
            return {
                'title': title,
                'game_name_slug': game_name_slug,
                'url': game_url,
                'metascore': metascore,
                'release_date': release_date,
                'discovered_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    ⚠ Error extracting game data: {e}")
            return None
    
    def discover_from_search(self, search_term, max_results=50):
        """
        Discover games from search results
        
        Args:
            search_term: Search query
            max_results: Maximum number of results to collect
        """
        print("=" * 70)
        print(f"GAME DISCOVERY - Search for: {search_term}")
        print("=" * 70)
        
        # Metacritic search URL
        search_url = f"{self.base_url}/search/{search_term}/game"
        
        print(f"\nFetching: {search_url}")
        
        try:
            response = self._fetch(search_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search results
            results = soup.select('div.search_results div.result')
            
            print(f"Found {len(results)} search results\n")
            
            for i, result in enumerate(results[:max_results], 1):
                game_data = self._extract_game_from_search_result(result)
                
                if game_data:
                    self.discovered_games.append(game_data)
                    print(f"[{i}] {game_data['title']} - Score: {game_data.get('metascore', 'N/A')}")
            
        except requests.RequestException as e:
            print(f"✗ Error searching: {e}")
        
        print("\n" + "=" * 70)
        print(f"DISCOVERY COMPLETE - Found {len(self.discovered_games)} games")
        print("=" * 70)
        
        return self.discovered_games
    
    def _extract_game_from_search_result(self, result):
        """Extract game information from a search result"""
        try:
            # Extract title and URL
            title_elem = result.select_one('h3.product_title a')
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            game_url = urljoin(self.base_url, title_elem.get('href', ''))
            
            # Extract game name from URL
            url_parts = [p for p in urlparse(game_url).path.split('/') if p]
            game_name_slug = url_parts[2] if len(url_parts) > 2 else url_parts[1] if len(url_parts) > 1 else 'unknown'
            
            # Extract metascore
            score_elem = result.select_one('div.metascore_w')
            metascore = score_elem.get_text(strip=True) if score_elem else None
            
            # Extract release date
            date_elem = result.select_one('span.release_date')
            release_date = date_elem.get_text(strip=True) if date_elem else None
            
            return {
                'title': title,
                'game_name_slug': game_name_slug,
                'url': game_url,
                'metascore': metascore,
                'release_date': release_date,
                'discovered_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"  ⚠ Error extracting search result: {e}")
            return None
    
    def save_to_file(self, output_format='json', filename=None):
        """Save discovered games to a file"""
        
        if not self.discovered_games:
            print("No games to save!")
            return
        
        # If incremental saving was used, file is already saved
        if self.incremental and self.output_file and self.output_file.exists():
            print(f"\n✓ Games already saved incrementally to: {self.output_file}")
            return self.output_file
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"discovered_games_{timestamp}.{output_format}"
        
        filepath = resolve_output_path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if output_format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.discovered_games, f, indent=2, ensure_ascii=False)
            
            elif output_format == 'csv':
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    if self.discovered_games:
                        writer = csv.DictWriter(f, fieldnames=self.discovered_games[0].keys())
                        writer.writeheader()
                        writer.writerows(self.discovered_games)
            
            elif output_format == 'txt':
                # Format: game_slug|platform_slug, one game per line
                with open(filepath, 'w', encoding='utf-8') as f:
                    for game in self.discovered_games:
                        platform_slug = game.get('platform_slug') or 'multiplatform'
                        f.write(f"{game['game_name_slug']}|{platform_slug}\n")
            
            print(f"\n✓ Saved {len(self.discovered_games)} games to: {filepath}")
            
        except Exception as e:
            print(f"✗ Error saving file: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Discover games from Metacritic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover all games (first 10 pages)
  python discover_games.py --max-pages 10
  
  # Discover PS5 games with score >= 80
  python discover_games.py --platform playstation-5 --min-score 80 --max-pages 5
  
  # Search for specific game
  python discover_games.py --search "god of war"
  
  # Save as CSV
  python discover_games.py --max-pages 5 --format csv
  
  # Split output into 3 files
  python discover_games.py --max-pages 10 --split-files 3
        """
    )
    
    parser.add_argument('--platform', type=str, default='all',
                      help='Platform filter (e.g., playstation-5, pc, switch, all)')
    parser.add_argument('--min-score', type=int, default=None,
                      help='Minimum metascore (0-100)')
    parser.add_argument('--max-pages', type=int, default=0,
                      help='Maximum number of pages to browse (default: 0, use 0 for unlimited)')
    parser.add_argument('--delay', type=float, default=1.0,
                      help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--format', type=str, default='txt',
                      choices=['json', 'csv', 'txt'],
                      help='Output format (default: txt)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output filename (default: auto-generated)')
    parser.add_argument('--search', type=str, default=None,
                      help='Search for specific game instead of browsing')
    parser.add_argument('--split-files', type=int, default=1,
                      help='Split output into N files (default: 1, no splitting)')
    parser.add_argument('--flaresolverr', type=str, default=os.getenv('FLARESOLVERR_URL'),
                      help='FlareSolverr URL to bypass Cloudflare (e.g. http://localhost:8191)')
    parser.add_argument('--no-auto-start-flaresolverr', dest='auto_start_flaresolverr',
                      action='store_false',
                      help='Không tự start container flaresolverr nếu container đang dừng')
    parser.set_defaults(auto_start_flaresolverr=True)
    
    args = parser.parse_args()
    
    # Create discoverer with incremental saving enabled by default
    discoverer = GameDiscoverer(
        delay=args.delay,
        max_pages=args.max_pages,
        incremental=True,
        split_files=args.split_files,
        flaresolverr_url=args.flaresolverr,
        auto_start_flaresolverr=args.auto_start_flaresolverr,
    )
    
    # Discover games
    if args.search:
        discoverer.discover_from_search(args.search)
    else:
        discoverer.discover_from_browse(
            platform=args.platform if args.platform != 'all' else None,
            min_score=args.min_score,
            output_format=args.format,
            filename=args.output
        )
    
    # Save results (will skip if already saved incrementally)
    if discoverer.discovered_games:
        discoverer.save_to_file(output_format=args.format, filename=args.output)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total games discovered: {len(discoverer.discovered_games)}")
        
        # Show score distribution
        score_ranges = {'90+': 0, '80-89': 0, '70-79': 0, '60-69': 0, '<60': 0, 'N/A': 0}
        for game in discoverer.discovered_games:
            score = game.get('metascore')
            if score:
                try:
                    score_val = int(score)
                    if score_val >= 90:
                        score_ranges['90+'] += 1
                    elif score_val >= 80:
                        score_ranges['80-89'] += 1
                    elif score_val >= 70:
                        score_ranges['70-79'] += 1
                    elif score_val >= 60:
                        score_ranges['60-69'] += 1
                    else:
                        score_ranges['<60'] += 1
                except (ValueError, TypeError):
                    score_ranges['N/A'] += 1
            else:
                score_ranges['N/A'] += 1
        
        print("\nScore distribution:")
        for range_name, count in score_ranges.items():
            if count > 0:
                print(f"  {range_name}: {count}")
        
        print("\n" + "=" * 70)
    else:
        print("\n⚠ No games discovered!")


if __name__ == "__main__":
    main()
