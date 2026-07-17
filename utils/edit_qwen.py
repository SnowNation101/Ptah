import requests
import threading
import queue
import base64
import os
from io import BytesIO
from PIL import Image
from concurrent.futures import Future
from typing import List, Optional, Any


class QwenEditError(Exception):
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception

class QwenEditClient:
    def __init__(
        self,
        server_urls: List[str],
        num_workers: Optional[int] = None,
        request_timeout: int = 120
    ):
        if not server_urls or not all(isinstance(url, str) for url in server_urls):
            raise ValueError("server_urls must be a non-empty list of strings.")
        
        self.server_urls = server_urls
        self.num_servers = len(server_urls)
        self.timeout = request_timeout
        
        if num_workers is None:
            num_workers = self.num_servers
        
        print(f"QwenEditClient initialized with {self.num_servers} server instances, {num_workers} internal workers.")

        self.task_queue = queue.Queue()
        self._workers = []
        self._active = True

        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"EditWorker-{i}")
            worker.daemon = True
            worker.start()
            self._workers.append(worker)

        self._lock = threading.Lock()
        self._server_index = 0
        self.session = requests.Session()

    def _get_next_server_url(self) -> str:
        with self._lock:
            url = self.server_urls[self._server_index]
            self._server_index = (self._server_index + 1) % self.num_servers
            return url

    def _worker_loop(self):
        while self._active:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                image, prompt, future, kwargs = task
                
                try:
                    result_image = self._send_request(image, prompt, **kwargs)
                    future.set_result(result_image)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
    
    def _send_request(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        server_url = self._get_next_server_url()
        endpoint = f"{server_url}/edit/"

        payload = {
            "image_b64": image_b64_string,
            "prompt": prompt,
            **kwargs
        }

        try:
            response = self.session.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("images"):
                img_b64 = data["images"][0]
                img_bytes = base64.b64decode(img_b64)
                return Image.open(BytesIO(img_bytes))
            else:
                raise QwenEditError(f"Fail to edit image: {data.get('detail', 'N/A')}")
        except requests.exceptions.RequestException as e:
            raise QwenEditError(f"Fail to edit image at {server_url}: {e}", original_exception=e)
        
    def edit_image(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        if not self._active:
            raise RuntimeError("QwenEditClient is not active.")

        future = Future()
        self.task_queue.put((image, prompt, future, kwargs))
        
        return future.result()

    def shutdown(self, wait: bool = True):
        if not self._active:
            return
            
        print("Shutdown QwenEditClient...")
        self._active = False
        for _ in self._workers:
            self.task_queue.put(None)
        
        if wait:
            for worker in self._workers:
                worker.join()
        print("QwenEditClient shutdown completed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class SiliconFlowQwenEditClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen-Image-Edit-2509",
        request_timeout: int = 120,
    ):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("SiliconFlow API key not provided (or missing in env var SILICONFLOW_API_KEY)")

        self.model = model
        self.timeout = request_timeout
        self.url = "https://api.siliconflow.cn/v1/images/generations"

        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        print("SiliconFlowQwenEditClient initialized.")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert a PIL.Image object to a base64 data URI string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64, {img_b64}"

    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        size: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Edit an image using SiliconFlow's Qwen Image Edit model.

        Args:
            image: Input image (PIL.Image)
            prompt: Editing instruction (e.g., "Draw a red arrow pointing to point D.")
            size: Optional output size, e.g. "1024x1024"
            seed: Optional random seed
            **kwargs: Additional parameters for the API

        Returns:
            PIL.Image: The edited image
        """
        img_data_uri = self._image_to_base64(image)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "image": img_data_uri,
        }

        if size:
            payload["size"] = size
        if seed is not None:
            payload["seed"] = seed
        payload.update(kwargs)

        try:
            response = self.session.post(
                self.url, json=payload, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            raise QwenEditError(f"Request failed: {e}", e)

        # Normal response handling
        if "images" in data and data["images"]:
            try:
                img_url = data["images"][0]["url"]
                img_bytes = self.session.get(img_url, timeout=self.timeout).content
                return Image.open(BytesIO(img_bytes))
            except Exception as e:
                raise QwenEditError(f"Failed to download image: {e}", e)

        # Error handling
        elif "message" in data:
            raise QwenEditError(f"Editing failed: {data['message']}")
        else:
            raise QwenEditError(f"Unexpected response: {data}")

    def __call__(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Enable direct calls: client(image, prompt)"""
        return self.edit_image(image, prompt, **kwargs)

    