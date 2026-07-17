import base64
import io
import os
import mimetypes
from pathlib import Path
from typing import Union

from PIL import Image


def encode_image_to_base64(
    image_input: Union[str, os.PathLike, Image.Image],
    max_size_mb: float = 20.0,
    check_is_valid_image: bool = True,
    pil_format: str | None = None,
) -> str:
    """
    Convert either a local image path or a PIL Image object to a Base64 Data URI.

    Args:
        image_input:
            - str / os.PathLike: path to a local image file
            - PIL.Image.Image: in-memory PIL image
        max_size_mb:
            Maximum allowed encoded source size in MB.
            - For file input: checks original file size
            - For PIL input: checks serialized byte size after saving to memory
        check_is_valid_image:
            - For file input: whether to verify image integrity with Pillow
            - For PIL input: whether to do a lightweight sanity check
        pil_format:
            Output format when input is a PIL Image, e.g. "PNG", "JPEG", "WEBP".
            If None:
            - try image.format
            - otherwise default to "PNG"

    Returns:
        A string like: 'data:image/png;base64,...'

    Raises:
        FileNotFoundError, IsADirectoryError, ValueError, IOError, TypeError
    """

    def _check_size_limit(num_bytes: int, label: str) -> None:
        size_mb = num_bytes / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"{label} size ({size_mb:.2f} MB) exceeds the limit of {max_size_mb} MB.")

    def _mime_from_format(fmt: str | None) -> str | None:
        if not fmt:
            return None
        mime = Image.MIME.get(fmt.upper())
        if mime:
            return mime

        fmt_lower = fmt.lower()
        if fmt_lower == "jpg":
            fmt_lower = "jpeg"
        return f"image/{fmt_lower}"

    # ------------------------------------------------------------------
    # Case 1: input is a local file path
    # ------------------------------------------------------------------
    if isinstance(image_input, (str, os.PathLike)):
        image_path = str(image_input)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file at path '{image_path}' does not exist.")
        if not os.path.isfile(image_path):
            raise IsADirectoryError(f"The path '{image_path}' points to a directory, not a file.")

        file_size_bytes = os.path.getsize(image_path)
        _check_size_limit(file_size_bytes, "Image")

        mime_type = None

        if check_is_valid_image:
            try:
                with Image.open(image_path) as img:
                    img.verify()
                    fmt = img.format
                    if not fmt:
                        mime_type = mimetypes.guess_type(image_path)[0]
                    else:
                        mime_type = _mime_from_format(fmt)
            except Exception as e:
                raise ValueError(
                    f"File '{image_path}' is not a valid image or is corrupted. Error: {str(e)}"
                )
        else:
            mime_type, _ = mimetypes.guess_type(image_path)

        if not mime_type or not mime_type.startswith("image/"):
            ext = Path(image_path).suffix.lower()
            known_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".tiff": "image/tiff",
            }
            mime_type = known_types.get(ext)

            if not mime_type:
                raise ValueError(f"Could not determine a valid image MIME type for '{image_path}'.")

        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        except OSError as e:
            raise IOError(f"Failed to read file contents: {e}")

        return f"data:{mime_type};base64,{encoded_string}"

    # ------------------------------------------------------------------
    # Case 2: input is a PIL Image
    # ------------------------------------------------------------------
    if isinstance(image_input, Image.Image):
        img = image_input

        if check_is_valid_image:
            if img.width <= 0 or img.height <= 0:
                raise ValueError("PIL image has invalid dimensions.")

        # Decide output format
        fmt = pil_format or img.format or "PNG"
        fmt = fmt.upper()

        # JPEG does not support alpha
        save_img = img
        if fmt in {"JPG", "JPEG"} and img.mode in {"RGBA", "LA", "P"}:
            save_img = img.convert("RGB")

        mime_type = _mime_from_format(fmt)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Could not determine a valid MIME type for PIL format '{fmt}'.")

        try:
            buffer = io.BytesIO()
            save_kwargs = {}

            if fmt in {"JPG", "JPEG"}:
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True

            save_img.save(buffer, format=fmt, **save_kwargs)
            raw_bytes = buffer.getvalue()
        except Exception as e:
            raise IOError(f"Failed to serialize PIL image: {e}")

        _check_size_limit(len(raw_bytes), "Serialized PIL image")

        encoded_string = base64.b64encode(raw_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"

    raise TypeError(
        "image_input must be a file path (str / os.PathLike) or a PIL.Image.Image instance."
    )