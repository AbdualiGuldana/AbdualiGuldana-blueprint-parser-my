import os
import subprocess
from pathlib import Path

DATA = "/workspace/AbdualiGuldana-blueprint-parser-my/Sample/01.원천데이터"
OUT = "/workspace/AbdualiGuldana-blueprint-parser-my/dots.ocr/output"
PARSER = "dots_ocr/parser.py"

for sub in ["OBJ", "OCR", "SPA", "STR"]:
    in_dir = Path(DATA) / sub
    out_dir = Path(OUT) / f"sample_{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for img in in_dir.glob("*.png"):  
        print(f"[{sub}] {img.name}")
        subprocess.run(["python3", PARSER, str(img), "--use_hf", "true", "--enable_metrics"])
        base = img.stem
        
        for ext in [".json", ".md", ".jpg"]:  
            src = Path(OUT) / f"{base}{ext}"
            if src.exists():
                src.rename(out_dir / src.name)

