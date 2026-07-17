import subprocess
import sys
from PIL import Image
import io
import re
import os


def sanitize_code(code: str) -> str:
    if not code:
        return ""

    code = re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())

    replacements = {
        "\\r\\n": "\n",
        "\\n": "\n",
        "\\r": "\n",
        "\\t": "\t",
        "\\\"": "\"",
        "\\'": "'",
        "\\\\": "\\",
    }
    
    for old, new in replacements.items():
        code = code.replace(old, new)

    def unicode_fix(match):
        try:
            return match.group(0).encode().decode('unicode-escape')
        except Exception:
            return match.group(0)
    
    code = re.sub(r'\\u[0-9a-fA-F]{4}', unicode_fix, code)

    bad_chars = {
        "\xa0": " ",
        "\u3000": "  ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
    for bad, good in bad_chars.items():
        code = code.replace(bad, good)

    return code.strip()

def get_font_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "..", "assets")
    
    simhei = os.path.join(assets_dir, "SimHei.ttf")
    arial = os.path.join(assets_dir, "Arial-Unicode-MS.ttf")
    
    return simhei, arial


def codeexec(llmcode):
    simhei_path, arial_path = get_font_paths()

    head_code = f"""
import sys
import io
import os
import matplotlib
import matplotlib.font_manager as fm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

for fpath in ['{simhei_path}', '{arial_path}']:
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']

"""

    footer_code = """\n\n
try:
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close('all')
    image_bytes = buffer.getvalue()
    sys.stdout.buffer.write(image_bytes)
    sys.stdout.flush()

except Exception as e:
    sys.stderr.write(f"fail: {{e}}")
    sys.stderr.flush()
    sys.exit(1)
    """

    try:
        llmcode = sanitize_code(llmcode)
        print("\n\n\n\n")
        print(llmcode)
        all_code = head_code + llmcode + footer_code

        completed_process = subprocess.run(
            [sys.executable, "-c", all_code],
            capture_output=True,
            text=False,
            check=True,
            timeout=15,
        )

        image_bytes_from_subprocess = completed_process.stdout

        if not image_bytes_from_subprocess:
            raise ValueError("No images returned")

        image = Image.open(io.BytesIO(image_bytes_from_subprocess))
        return image, ""

    except subprocess.TimeoutExpired:
        print("[WARNING] Code execution timeout.")
        return None, "Code execution timeout."
    except subprocess.CalledProcessError as e:
        print("[WARNING] Code execution failed.")
        print("Error message:")
        print(e.stderr)
        return None, f"Code execution failed with error message: {e.stderr}"
    except Exception as e:
        print(f"[WARNING] Code execution failed with error message: {e}")
        return None, f"Code execution failed with error message: {e}"
