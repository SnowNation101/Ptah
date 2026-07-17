import re
import os

from PIL.Image import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.code_exec import codeexec
from utils.search_client import SerpSearchClient, SerperSearchClient
from utils.generation_qwen import QwenImageClient, SiliconFlowQwenImageClient
from utils.edit_qwen import QwenEditClient, SiliconFlowQwenEditClient
from typing import List, Optional, Dict, Any
import ast


class ImageTools:
    def __init__(
        self,
        diffusion_backbone: str = "sf_qwen",
        edit_backbone: str = "sf_qwen",
        search_provider: str = "serper",
        search_topk: int = 15,
        qwen_image_server_urls: List[str] = None,
        qwen_edit_server_urls: List[str] = None,
    ):
        if diffusion_backbone == "seed":
            from utils.generation_seed import SeedreamClient

            self.diffusion_client = SeedreamClient()
        elif diffusion_backbone == "qwen":
            if not qwen_image_server_urls:
                qwen_image_server_urls = [os.getenv("QWEN_IMAGE_SERVER_URL")]
            self.diffusion_client = QwenImageClient(qwen_image_server_urls)
        elif diffusion_backbone == "sf_qwen":
            self.diffusion_client = SiliconFlowQwenImageClient()
        else:
            raise ValueError(f"Not Implemented diffusion client: {diffusion_backbone}.")

        if edit_backbone == "seed":
            from utils.edit_seed import SeedEditClient

            self.edit_client = SeedEditClient()
        elif edit_backbone == "qwen":
            if not qwen_edit_server_urls:
                qwen_edit_server_urls = [os.getenv("QWEN_EDIT_SERVER_URL")]
            self.edit_client = QwenEditClient(qwen_edit_server_urls)
        elif edit_backbone == "sf_qwen":
            self.edit_client = SiliconFlowQwenEditClient()
        else:
            raise ValueError(f"Not Implemented edit client: {edit_backbone}.")

        if search_provider == "serp":
            self.search_client = SerpSearchClient(topk=search_topk)
        elif search_provider == "serper":
            self.search_client = SerperSearchClient(topk=search_topk)
        else:
            raise ValueError(f"Not Implemented search client: {search_provider}.")

    def tag2image(
        self, imagetag: str, tag_len: int, multimodal_inputs: list[Image] = None
    ) -> tuple[Image | None, str | None]:
        if not multimodal_inputs:
            mm_len = 0
        else:
            mm_len = len(multimodal_inputs)

        if not self._is_image_valid(imagetag, tag_len, mm_len):
            return None, None

        data = self._parse_image_tag(imagetag)
        source = data["source"]

        if source == "diffusion":
            prompt = data["params"]["prompt"]
            print("Generating image with diffusion prompt:", prompt)
            image = self.diffusion_client.get_pil_image(prompt)
        elif source == "search":
            prompt = data["params"]["query"]
            print("Searching image with query:", prompt)
            image = self.search_client.search_image(prompt)
        elif source == "code":
            code = data["params"]["code"]
            print("Generating image with code execution.")
            image, _ = codeexec(code)
        elif source == "ref":
            if not multimodal_inputs:
                image = None
            else:
                img_num = data["params"]["img_index"]
                print(f"Referencing image index {img_num}")
                try:
                    image = multimodal_inputs[int(img_num)]
                except Exception as e:
                    print(f"Error referencing image: {e}")
                    image = None
        else:  # edit
            if not multimodal_inputs:
                image = None
            else:
                prompt = data["params"]["prompt"]
                img_num = data["params"]["img_index"]
                print(f"Editing image index {img_num} with prompt:", prompt)
                try:
                    img = multimodal_inputs[int(img_num)].copy()
                    image = self.edit_client.edit_image(img, prompt)
                except Exception as e:
                    print(f"Error editing image: {e}")
                    image = None
        return image, source

    @staticmethod
    def _find_imgen_blocks(content: str) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        n = len(content)
        i = 0
        lower = content.lower()

        while i < n:
            open_idx = lower.find("<imgen", i)
            if open_idx == -1:
                break

            gt = content.find(">", open_idx)
            if gt == -1:
                break

            inner_start = gt + 1

            open_tag_text = content[open_idx:gt + 1]
            if open_tag_text.rstrip().endswith("/>"):
                blocks.append({
                    "span": (open_idx, gt + 1),
                    "payload": "",
                    "full_tag": open_tag_text,
                })
                i = gt + 1
                continue

            close_tag = "</imgen>"
            close_idx = lower.find(close_tag, inner_start)
            if close_idx == -1:
                break

            end_idx = close_idx + len(close_tag)
            full_tag = content[open_idx:end_idx]
            inner = content[inner_start:close_idx]

            left = inner.find("{")
            right = inner.rfind("}")
            if left != -1 and right != -1 and right > left:
                payload = inner[left:right + 1].strip()
            else:
                payload = inner.strip()

            payload = re.sub(r"\s*/\s*\Z", "", payload)

            blocks.append({
                "span": (open_idx, end_idx),
                "payload": payload,
                "full_tag": full_tag,
            })

            i = end_idx

        return blocks

    @staticmethod
    def find_imgen_tags(content: str) -> list[str]:
        return [b["payload"] for b in ImageTools._find_imgen_blocks(content)]

    def generate(
        self, content: str, multimodal_inputs: list[Image] = None, max_workers: int = 8
    ) -> tuple[str, list[Image], dict]:
        
        blocks = self._find_imgen_blocks(content)
        image_tags_info = [b["payload"] for b in blocks]

        if not image_tags_info:
            empty_stats = {
                "diffusion": {"total": 0, "success": 0, "success_rate": 1.0},
                "search": {"total": 0, "success": 0, "success_rate": 1.0},
                "code": {"total": 0, "success": 0, "success_rate": 1.0},
                "edit": {"total": 0, "success": 0, "success_rate": 1.0},
                "ref": {"total": 0, "success": 0, "success_rate": 1.0},
            }
            return content, [], empty_stats

        initial_mm_len = len(multimodal_inputs) if multimodal_inputs else 0
        total_tags = len(image_tags_info)
        print(f"Total image tags to process: {total_tags}")
        
        jobs = []
        for i, tag_info in enumerate(image_tags_info):
            print("Processing image tag:", tag_info)

            data = self._parse_image_tag(tag_info)

            if not isinstance(data, dict) or "source" not in data:
                job = {
                    "tag_info": tag_info,
                    "source": None,
                    "params": {},
                    "original_index": i,
                    "dependency": float("inf"),
                }
                jobs.append(job)
                continue

            job = {
                "tag_info": tag_info,
                "source": data["source"],
                "params": data.get("params", {}),
                "original_index": i,
                "dependency": None,
            }

            # Handle dependency for both 'edit' and 'ref'
            if data["source"] in ["edit", "ref"]:
                try:
                    img_index = int(data["params"].get("img_index", -1))
                    total_possible_images = initial_mm_len + total_tags

                    if not (0 <= img_index < total_possible_images):
                        job["dependency"] = float("inf")
                    elif img_index >= initial_mm_len:
                        dependency_on_job_index = img_index - initial_mm_len
                        # Prevent forward dependency or self dependency
                        if dependency_on_job_index >= i:
                            job["dependency"] = float("inf")
                        else:
                            job["dependency"] = dependency_on_job_index
                except Exception as e:
                    print(f"Error parsing img_index for {data['source']} job: {e}")
                    job["dependency"] = float("inf")

            jobs.append(job)

        results = [None] * len(jobs)
        pending_jobs_indices = set(range(len(jobs)))

        while pending_jobs_indices:
            current_new_images = [
                res[0] if res is not None else None for res in results
            ]
            live_image_pool = (
                multimodal_inputs if multimodal_inputs else []
            ) + current_new_images

            runnable_batch_indices = []
            for idx in pending_jobs_indices:
                dep = jobs[idx]["dependency"]
                is_runnable = False
                if dep is None:
                    is_runnable = True
                elif isinstance(dep, int):
                    # Check if the dependent job has finished
                    if results[dep] is not None:
                        is_runnable = True
                if is_runnable:
                    runnable_batch_indices.append(idx)

            if not runnable_batch_indices:
                # Deadlock or broken dependencies; fail remaining
                for idx in pending_jobs_indices:
                    results[idx] = (None, jobs[idx]["source"])
                break

            batch_results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self.tag2image,
                        jobs[idx]["tag_info"],
                        total_tags,
                        live_image_pool,
                    ): idx
                    for idx in runnable_batch_indices
                }
                for future in as_completed(future_to_index):
                    original_index = future_to_index[future]
                    try:
                        batch_results[original_index] = future.result()
                    except Exception as exc:
                        print(f"Job {original_index} generated an exception: {exc}")
                        batch_results[original_index] = (
                            None,
                            jobs[original_index]["source"],
                        )

            for index, result_tuple in batch_results.items():
                results[index] = result_tuple
                pending_jobs_indices.remove(index)

        stats = {
            "diffusion": {"total": 0, "success": 0},
            "search": {"total": 0, "success": 0},
            "code": {"total": 0, "success": 0},
            "edit": {"total": 0, "success": 0},
            "ref": {"total": 0, "success": 0}, # Added ref stats
        }
        pil_images_list = []
        for image, source in results:
            if source is None:
                continue
            # Ensure source exists in stats (handles unknown sources gracefully-ish)
            if source not in stats:
                stats[source] = {"total": 0, "success": 0}
                
            stats[source]["total"] += 1
            if isinstance(image, Image):
                pil_images_list.append(image)
                stats[source]["success"] += 1

        processed_parts = []
        cursor = 0
        for block, (image, _) in zip(blocks, results):
            start, end = block["span"]
            processed_parts.append(content[cursor:start])
            processed_parts.append("<image>" if isinstance(image, Image) else "<fail_to_generate_image>")
            cursor = end
        processed_parts.append(content[cursor:])
        processed_text = "".join(processed_parts)

        stats_with_rate = {}
        for source, data in stats.items():
            total, success = data["total"], data["success"]
            stats_with_rate[source] = {
                "total": total,
                "success": success,
                "success_rate": round(success / total, 4) if total > 0 else 1.0,
            }

        return processed_text, pil_images_list, stats_with_rate

    @staticmethod
    def _parse_image_tag(dict_string: str) -> Optional[Dict[str, Any]]:
        if "\"source\": \"code\"" in dict_string:
            code_pattern = r"^\s*{\"source\": \"code\", \"description\": \"([^\"]*)\", \"params\": {\"code\": \"(.*)\"}}\s*$"
            match = re.search(code_pattern, dict_string, re.DOTALL)   
            if match:
                description, code = match.groups()
                return {
                    "source": "code",
                    "description": description,
                    "params": {"code": code}
                }
        try:
            parsed_dict = ast.literal_eval(dict_string)
            if isinstance(parsed_dict, dict):
                return parsed_dict
            return None
        except Exception:
            return None


    def _is_image_valid(self, content: str, tag_len: int, input_image_num: int = 0) -> bool:
        data = self._parse_image_tag(content)

        if not data:
            return False
        if not isinstance(data, dict):
            return False
        required_keys = {"source", "description", "params"}
        if set(data.keys()) != required_keys:
            return False
        if not isinstance(data.get("description"), str) or not isinstance(
            data.get("params"), dict
        ):
            return False
        source = data.get("source")
        params = data.get("params")
        if source == "diffusion":
            return isinstance(params.get("prompt"), str) and set(params.keys()) == {
                "prompt"
            }
        elif source == "search":
            return isinstance(params.get("query"), str) and set(params.keys()) == {"query"}
        elif source == "code":
            return isinstance(params.get("code"), str) and set(params.keys()) == {"code"}
        elif source == "edit":
            return (
                isinstance(params.get("img_index"), int)
                and (params.get("img_index") >= 0)
                and params.get("img_index") < input_image_num + tag_len
                and isinstance(params.get("prompt"), str)
                and set(params.keys()) == {"img_index", "prompt"}
            )
        elif source == "ref":
            return (
                isinstance(params.get("img_index"), int)
                and (params.get("img_index") >= 0)
                and params.get("img_index") < input_image_num + tag_len
                and set(params.keys()) == {"img_index"}
            )
        return False
