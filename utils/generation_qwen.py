import requests
import threading
import queue
import base64
from io import BytesIO
from PIL import Image
from concurrent.futures import Future
from typing import List, Optional, Any

from dotenv import load_dotenv
load_dotenv()

class QwenImageGenerationError(Exception):
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception


class QwenImageClient:
    def __init__(
        self,
        server_urls: List[str],
        num_workers: Optional[int] = None,
        request_timeout: int = 120
    ):

        if not server_urls or not all(isinstance(url, str) for url in server_urls):
            raise ValueError("server_urls must be a url list")
        
        self.server_urls = server_urls
        self.num_servers = len(server_urls)
        self.timeout = request_timeout
        
        if num_workers is None:
            num_workers = self.num_servers
        
        print(f"Client initialized, connected to {self.num_servers} server instances, {num_workers} internal worker threads started.")

        self.task_queue = queue.Queue()
        self._workers = []
        self._active = True

        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"Worker-{i}")
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
                
                prompt, future, kwargs = task
                
                try:
                    result_image = self._send_request(prompt, **kwargs)
                    future.set_result(result_image)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
    
    def _send_request(self, prompt: str, **kwargs: Any) -> Image.Image:
        server_url = self._get_next_server_url()
        endpoint = f"{server_url}/generate/"
        payload = {"prompt": prompt, **kwargs}

        try:
            response = self.session.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("images"):
                img_b64 = data["images"][0]
                img_bytes = base64.b64decode(img_b64)
                return Image.open(BytesIO(img_bytes))
            else:
                raise QwenImageGenerationError(f"Fail to generate image: {data.get('detail', 'N/A')}")
        except requests.exceptions.RequestException as e:
            raise QwenImageGenerationError(f"Fail to request {server_url}", original_exception=e)
        
    def get_pil_image(self, prompt: str, **kwargs: Any) -> Image.Image:
        if not self._active:
            raise RuntimeError("Client is closed, cannot submit new tasks.")

        future = Future()
        self.task_queue.put((prompt, future, kwargs))
        
        return future.result()

    def shutdown(self, wait: bool = True):
        if not self._active:
            return
            
        print("Shutting down client...")
        self._active = False
        for _ in self._workers:
            self.task_queue.put(None)
        
        if wait:
            for worker in self._workers:
                worker.join()
        print("Client shutdown complete.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class SiliconFlowQwenImageClient:
    """
    A lightweight client that uses SiliconFlow's public API to generate images
    via the Qwen/Qwen-Image model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_timeout: int = 120,
        api_base: str = "https://api.siliconflow.cn/v1/images/generations"
    ):
        """
        :param api_key: SiliconFlow API key (can also be read from env var SILICONFLOW_API_KEY)
        :param request_timeout: request timeout (seconds)
        :param api_base: API base URL for SiliconFlow
        """
        import os

        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("SiliconFlow API key not provided (or missing in env var SILICONFLOW_API_KEY)")

        self.api_base = api_base
        self.timeout = request_timeout
        self.session = requests.Session()

        print("SiliconFlowQwenImageClient initialized.")

    def get_pil_image(self, prompt: str, **kwargs: Any) -> Image.Image:
        """
        Generate an image via SiliconFlow's Qwen/Qwen-Image API.

        :param prompt: text prompt for image generation
        :param kwargs: additional fields (e.g., model, size, seed, etc.)
        :return: PIL.Image object
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": kwargs.pop("model", "Qwen/Qwen-Image"),
            "prompt": prompt,
        }
        payload.update(kwargs)

        try:
            response = self.session.post(
                self.api_base, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if "images" in data and data["images"]:
                image_url = data["images"][0].get("url")
                if not image_url:
                    raise QwenImageGenerationError("No image URL found in response")

                img_response = self.session.get(image_url, timeout=self.timeout)
                img_response.raise_for_status()
                return Image.open(BytesIO(img_response.content))

            else:
                raise QwenImageGenerationError(
                    f"Invalid API response: {data}"
                )

        except requests.exceptions.RequestException as e:
            raise QwenImageGenerationError(
                f"SiliconFlow API request failed: {e}", original_exception=e
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        print("SiliconFlow client closed.")


if __name__ == "__main__":
    client = SiliconFlowQwenImageClient()
    prompt = "A futuristic cityscape at sunset, with flying cars and neon lights, in the style of cyberpunk art."
    image = client.get_pil_image(prompt)
    image.save("generated_image.png")