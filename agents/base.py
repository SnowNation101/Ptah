import re
import json
from core.model_client import ModelClient

class BaseAgent:
    def __init__(self, client: ModelClient, system_prompt_path: str):
        self.client = client
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def _extract_json(self, text: str, start_tag: str, end_tag: str):
        if start_tag in text and end_tag in text:
            block = text.split(start_tag)[-1].split(end_tag)[0].strip()
            try:
                return json.loads(block)
            except:
                return None
        return None