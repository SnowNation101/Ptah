import os
import base64
import mimetypes
import re
from bs4 import BeautifulSoup, NavigableString, Tag

def encode_image_to_base64(image_path_or_url, base_path="."):
    """
    Resolve an image reference:
    1. Return http, https, and data URLs unchanged.
    2. Convert local files to base64 data URLs.
    """
    if not image_path_or_url:
        return None

    # Return existing data or remote URLs unchanged.
    if image_path_or_url.startswith(('http://', 'https://', 'data:')):
        return image_path_or_url

    # Reject malformed long strings that are unlikely to be filesystem paths.
    if len(image_path_or_url) > 4096:
        print(f"Warning: Image src length ({len(image_path_or_url)}) is too long to be a file path and missing 'data:' prefix. Skipping.")
        return None

    # Treat the value as a local file path.
    full_path = os.path.join(base_path, image_path_or_url)
    
    if not os.path.exists(full_path):
        # Warn only for path-like values to avoid noisy logs.
        if len(image_path_or_url) < 256:
            print(f"Warning: Image not found at {full_path}, skipping image.")
        return None

    # Infer MIME type.
    mime_type, _ = mimetypes.guess_type(full_path)
    if not mime_type:
        mime_type = 'image/jpeg' # Default fallback.

    try:
        with open(full_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error reading image file {full_path}: {e}")
        return None

def html_to_message_content(html_content, base_image_path="."):
    """
    Convert an HTML string into OpenAI-compatible multimodal message content.
    
    Args:
        html_content (str): Raw HTML content.
        base_image_path (str): Base directory for relative image paths.
        
    Returns:
        List: [...]
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')

    # Prefer the main report container, then body, then the full document.
    container = soup.find(class_='container') or soup.find('body') or soup

    content_list = []
    current_text_buffer = []

    def flush_text_buffer():
        """Append buffered text as one content item."""
        if current_text_buffer:
            text_content = "\n\n".join(current_text_buffer).strip()
            if text_content:
                content_list.append({
                    "type": "text", 
                    "text": text_content
                })
            current_text_buffer.clear()

    # Preserve basic markdown-like structure while walking the DOM.
    def process_element(element):
        # Ignore non-content tags.
        if element.name in ['script', 'style', 'meta', 'title', 'link']:
            return

        # Convert images to image_url items.
        if element.name == 'img':
            src = element.get('src')
            if src:
                # Flush text before inserting an image.
                flush_text_buffer()
                
                # Resolve image data.
                image_url = encode_image_to_base64(src, base_image_path)
                if image_url:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    })
            return

        # Preserve text nodes.
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                current_text_buffer.append(text)
            return

        # Preserve simple semantic wrappers.
        prefix = ""
        suffix = ""
        
        if element.name == 'h1': prefix, suffix = "# ", ""
        elif element.name == 'h2': prefix, suffix = "## ", ""
        elif element.name == 'h3': prefix, suffix = "### ", ""
        elif element.name == 'li': prefix, suffix = "- ", ""
        elif element.name == 'strong' or element.name == 'b': prefix, suffix = "**", "**"
        elif element.name == 'p': suffix = "" # Paragraph breaks come from buffer joins.

        # Recurse when nested tags are present, including mixed text/image nodes.
        has_child_elements = any(isinstance(c, Tag) for c in element.children)
        
        if not has_child_elements:
            # Directly extract plain text leaf nodes.
            text = element.get_text(strip=True)
            if text:
                current_text_buffer.append(f"{prefix}{text}{suffix}")
        else:
            # Recurse through child tags.
            if prefix: current_text_buffer.append(prefix)
            for child in element.children:
                process_element(child)
            if suffix: current_text_buffer.append(suffix)

    # Traverse top-level children.
    if hasattr(container, 'children'):
        for child in container.children:
            process_element(child)
    else:
        # Fallback if container has no children (e.g. string only)
        process_element(container)

    # Flush trailing text.
    flush_text_buffer()

    return content_list

if __name__ == "__main__":
    base64_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    html_sample = f"""
    <!DOCTYPE html>
    <html lang="en">
    <body>
    <div class="container">
        <h1>Report with Base64 Images</h1>
        <p>This is the first paragraph.</p>
        
        <img src="{base64_img}" alt="Red Dot">
        
        <h2>Section 2</h2>
        <p>Text after image.</p>
    </div>
    </body>
    </html>
    """

    message = html_to_message_content(html_sample)

    import json
    print(json.dumps(message, indent=2, ensure_ascii=False))
