import requests
from PIL import Image, UnidentifiedImageError
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import os
import json


class SerpSearchClient:

    def __init__(
        self,
        api_key: str = None,
        url: str = "https://serpapi.com/search",
        topk: int = 15,
        engine: str = "google_images",
        timeout: int = 2,
        user_agent: str = "Mozilla/5.0",
        retry_times: int = 3,
    ):
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("SERP_API_KEY")
        self.url = url
        self.topk = topk
        self.engine = engine
        self.timeout = timeout
        self.user_agent = user_agent
        self.MIN_RESOLUTION = 400 * 400
        self.MAX_RESOLUTION = 1024 * 1024 * 5 * 5
        self.retry_times = retry_times
        self._validate_connection()
    
    def search_multimedia(self, query: str) -> List[str]:
        search_params = {
            "q": query,
            "engine": self.engine,
            "api_key": self.api_key,
        }

        response = requests.get(self.url, params=search_params)

        if response.ok and "error" not in response.text:
            data = response.json()
            result = data.get("images_results", [])[:self.topk]
        else:
            result = None
        return result


    def _validate_connection(self) -> None:
        res = self.search_multimedia(
            query="apple inc",
        )
        if res:
            print(
                f"Search Client Init Success, topk={self.topk}"
            )
        else:
            print("[WARNING] Failed to initialize SearchClient.")

    def _search(self, prompt: str, try_time: int) -> List[str]:
        res = self.search_multimedia(
            query=prompt,
        )
        if res:
            urls = []
            for item in res.result:
                urls.append(item["url"])
            return urls
        else:
            print(f"Search Error: {res.status}, try_time: {try_time}")
            return []

    def _convert_to_image(self, image_bytes: bytes) -> Image.Image | None:
        image_buffer = io.BytesIO(image_bytes)
        with Image.open(image_buffer) as img:
            img.verify()
        image_buffer.seek(0)
        pil_image = Image.open(image_buffer)
        img_copy = pil_image.copy()
        pil_image.close()
        return img_copy

    def _download_image(self, url: str) -> Image.Image | None:
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            pil_image = self._convert_to_image(response.content)

            width, height = pil_image.size
            resolution = width * height
            if self.MIN_RESOLUTION <= resolution <= self.MAX_RESOLUTION:
                return pil_image
            return None

        except requests.exceptions.RequestException as e:
            pass
        except UnidentifiedImageError:
            pass
        except Exception as e:
            print(f"[ERROR] {e}")
        return None

    def get_pil_image(self, prompt: str) -> Optional[Image.Image]:
        try:
            for t in range(self.retry_times):
                url_list = self._search(prompt, t)
                if url_list:
                    break
        except:
            return None

        if not url_list:
            print("Search Error: No URLs found.")
            return None
        with ThreadPoolExecutor(max_workers=self.topk) as executor:
            future_to_url = {
                executor.submit(self._download_image, url): url for url in url_list
            }
            futures_in_order = list(future_to_url.keys())

            for i, future in enumerate(futures_in_order):
                try:
                    pil_image = future.result()
                    if pil_image:
                        for f in futures_in_order[i + 1 :]:
                            f.cancel()
                        return pil_image
                except Exception as e:
                    print(f"[ERROR] {e}")

        print(f"[WARNING] All {len(url_list)} URLs failed. Search Result: None")
        return None


class SerperSearchClient:
    """
    A client for fetching both text and image search results using Serper.dev API.
    """

    def __init__(
        self,
        api_key: str = None,
        topk: int = 15,
        timeout: int = 2,
        user_agent: str = "Mozilla/5.0",
        retry_times: int = 3,
    ):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.url_image = "https://google.serper.dev/images"
        self.url_text = "https://google.serper.dev/search"
        self.topk = topk
        self.timeout = timeout
        self.user_agent = user_agent
        self.retry_times = retry_times

        # Resolution thresholds
        self.MIN_RESOLUTION = 400 * 400
        self.MAX_RESOLUTION = 1024 * 1024 * 5 * 5

        print("SerperSearchClient initialized.")

    def search_image(self, query: str) -> Optional[Image.Image]:
        """
        Perform an image search and return the first valid PIL image.
        This is the public API for image fetching.
        """
        for t in range(self.retry_times):
            url_list = self._fetch_image_urls(query, t)
            if url_list:
                pil_image = self._download_first_valid_image(url_list)
                if pil_image:
                    return pil_image

        print(f"[WARNING] No valid image found for query '{query}'.")
        return None

    def _fetch_image_urls(self, query: str, try_time: int) -> List[str]:
        """Search and extract image URLs from Serper.dev API."""
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = json.dumps({"q": query})
        try:
            response = requests.post(
                self.url_image, headers=headers, data=payload, timeout=self.timeout
            )
            if not response.ok:
                print(f"[ERROR] Image search failed: {response.status_code}")
                return []

            data = response.json()
            results = data.get("images", [])[:self.topk]
            return [item["imageUrl"] for item in results if "imageUrl" in item]

        except Exception as e:
            print(f"[ERROR] _fetch_image_urls failed: {e}, try_time={try_time}")
            return []

    def _download_first_valid_image(self, url_list: List[str]) -> Optional[Image.Image]:
        """Try downloading images concurrently and return the first valid one."""
        with ThreadPoolExecutor(max_workers=self.topk) as executor:
            future_to_url = {
                executor.submit(self._download_image, url): url for url in url_list
            }
            for future in future_to_url:
                try:
                    img = future.result()
                    if img:
                        # Cancel remaining tasks
                        for f in future_to_url:
                            f.cancel()
                        return img
                except Exception as e:
                    print(f"[ERROR] {e}")
        return None

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download and validate an image from a given URL."""
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, timeout=self.timeout, headers=headers)
            response.raise_for_status()

            img = self._convert_to_image(response.content)
            if not img:
                return None

            width, height = img.size
            resolution = width * height
            if self.MIN_RESOLUTION <= resolution <= self.MAX_RESOLUTION:
                return img
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            print(f"[ERROR] {e}")
        return None

    def _convert_to_image(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Convert downloaded bytes into a verified PIL image."""
        try:
            image_buffer = io.BytesIO(image_bytes)
            with Image.open(image_buffer) as img:
                img.verify()
            image_buffer.seek(0)
            pil_image = Image.open(image_buffer)
            copy = pil_image.copy()
            pil_image.close()
            return copy
        except UnidentifiedImageError:
            return None
        
