# src/utils/kfp_utils.py

from pathlib import Path
from kfp.components import load_component_from_text

def load_component_from_text_utf8(yaml_path: Path):
    """Carga un componente YAML preservando UTF-8 (Windows-safe)."""
    yaml_text = yaml_path.read_text(encoding="utf-8")
    return load_component_from_text(yaml_text)
