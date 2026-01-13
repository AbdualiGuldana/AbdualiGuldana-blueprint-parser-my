#pip install opencv-python pillow numpy

#install korean fonts in server (Ubuntu/Linux)
#sudo apt-get update
#sudo apt-get install -y fonts-nanum

#Usage -> python ground_truth_visualization.py SPA (or STR, OBJ, OCR)

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont

# Path
BASE_IMAGE_PATH = Path("/workspace/AbdualiGuldana-blueprint-parser-my/Sample/01.원천데이터")  # Folder containing SPA, STR, OBJ, OCR subfolders
BASE_JSON_PATH = Path("/workspace/AbdualiGuldana-blueprint-parser-my/Sample/02.라벨링데이터")     # Folder containing SPA, STR, OBJ, OCR subfolders
BASE_OUTPUT_PATH = Path("/workspace/AbdualiGuldana-blueprint-parser-my/ground_truth_visualization_output") # Output folder


# Korean font path (Linux/Ubuntu)
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

DEFAULT_COLOR = (0, 0, 255)  # Red
STR_COLORS = {k: DEFAULT_COLOR for k in [9, 10, 11]}
SPA_COLORS = {k: DEFAULT_COLOR for k in [1, 2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]}
OBJ_COLORS = {k: DEFAULT_COLOR for k in [4, 5, 6, 7, 8]}

# Category names
STR_NAMES = {9: "출입문", 10: "창호", 11: "벽체"}
SPA_NAMES = {
    1: "다목적공간", 2: "엘리베이터홀", 3: "계단실", 13: "거실",
    14: "침실", 15: "주방", 16: "현관", 17: "발코니", 18: "화장실",
    19: "실외기실", 20: "드레스룸", 22: "기타", 23: "엘리베이터"
}
OBJ_NAMES = {4: "변기", 5: "세면대", 6: "싱크대", 7: "욕조", 8: "가스레인지"}

def get_korean_font(size=20):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception as e:
        print(f"Warning: Korean font not found at {FONT_PATH}")
        return ImageFont.load_default()

def put_korean_text(img, text, position, font_size=20, color=(255, 255, 255), bg_color=None):
    """Add Korean text to CV2 image using PIL."""
    font = get_korean_font(font_size)
    
    # Convert CV2 BGR to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Convert BGR to RGB for PIL
    text_color_rgb = (color[2], color[1], color[0])
    
    # Draw background if specified
    if bg_color is not None:
        bbox = draw.textbbox(position, text, font=font)
        padding = 5
        bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, 
             bbox[2] + padding, bbox[3] + padding],
            fill=bg_color_rgb
        )
    
    # Draw text
    draw.text(position, text, font=font, fill=text_color_rgb)
    
    # Convert PIL RGB back to CV2 BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_legend(img, legend_items, position=(10, 30)):
    """Draw legend with Korean text."""
    y = position[1]
    for color_bgr, label in legend_items:
        # Draw colored rectangle with CV2
        cv2.rectangle(img, (position[0], y - 15), (position[0] + 25, y + 5), color_bgr, -1)
        cv2.rectangle(img, (position[0], y - 15), (position[0] + 25, y + 5), (255, 255, 255), 1)
        
        # Add text with PIL
        img = put_korean_text(img, label, (position[0] + 35, y - 15), 
                             font_size=20, color=(255, 255, 255))
        y += 35
    
    return img

def visualize_ocr(image_path, json_path, output_path):
    """Visualize OCR annotations."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    color = (255, 0, 0)  # Blue in BGR
    
    for ann in data['annotations']:
        if ann['category_id'] == 21:  # OCR category
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            ocr_text = ann['attributes'].get('OCR', '')
            
            # Draw rectangle with CV2
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Add Korean text with PIL
            img = put_korean_text(img, ocr_text, (x + 2, y - 30), 
                                 font_size=20, color=(255, 255, 255), bg_color=color)
    
    cv2.imwrite(str(output_path), img)
    return True

def visualize_obj(image_path, json_path, output_path):
    """Visualize object annotations."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for ann in data['annotations']:
        cat_id = ann['category_id']
        
        if cat_id not in OBJ_COLORS:
            continue
        
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        rotation = ann['attributes'].get('rotation', 0.0)
        color = OBJ_COLORS[cat_id]
        label = OBJ_NAMES[cat_id]
        
        # Draw rectangle with CV2
        if rotation != 0.0:
            # Draw rotated rectangle
            cx, cy = x + w/2, y + h/2
            angle_rad = np.deg2rad(rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = corners @ rotation_matrix.T
            rotated[:, 0] += cx
            rotated[:, 1] += cy
            
            cv2.polylines(img, [rotated.astype(np.int32)], True, color, 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Add Korean label
        img = put_korean_text(img, label, (x + 2, y - 35), 
                             font_size=24, color=(255, 255, 255), bg_color=color)
    
    cv2.imwrite(str(output_path), img)
    return True

def visualize_str(image_path, json_path, output_path):
    """Visualize structure annotations with semantic segmentation."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    # Fill polygons with CV2
    for ann in data['annotations']:
        cat_id = ann['category_id']
        
        if cat_id not in STR_COLORS:
            continue
        
        color = STR_COLORS[cat_id]
        
        if ann['segmentation'] and len(ann['segmentation']) > 0:
            for seg in ann['segmentation']:
                if len(seg) >= 6:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], color)
    
    # Blend
    overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)
    
    # Draw contours
    for ann in data['annotations']:
        cat_id = ann['category_id']
        
        if cat_id not in STR_COLORS:
            continue
        
        color = STR_COLORS[cat_id]
        
        if ann['segmentation'] and len(ann['segmentation']) > 0:
            for seg in ann['segmentation']:
                if len(seg) >= 6:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(overlay, [points], True, color, 2)
    
    # Add legend
    legend_items = [(color, STR_NAMES[cat_id]) for cat_id, color in STR_COLORS.items()]
    overlay = draw_legend(overlay, legend_items)
    
    cv2.imwrite(str(output_path), overlay)
    return True

def visualize_spa(image_path, json_path, output_path):
    """Visualize space annotations with semantic segmentation."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    # Fill polygons
    for ann in data['annotations']:
        cat_id = ann['category_id']
        
        if cat_id not in SPA_COLORS:
            continue
        
        color = SPA_COLORS[cat_id]
        
        if ann['segmentation'] and len(ann['segmentation']) > 0:
            for seg in ann['segmentation']:
                if len(seg) >= 6:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], color)
    
    # Blend
    overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    
    # Draw contours and labels
    for ann in data['annotations']:
        cat_id = ann['category_id']
        
        if cat_id not in SPA_COLORS:
            continue
        
        color = SPA_COLORS[cat_id]
        
        if ann['segmentation'] and len(ann['segmentation']) > 0:
            for seg in ann['segmentation']:
                if len(seg) >= 6:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(overlay, [points], True, color, 2)
                    
                    # Add label at centroid
                    M = cv2.moments(points)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        label = SPA_NAMES.get(cat_id, "")
                        overlay = put_korean_text(overlay, label, (cx - 30, cy - 10), 
                                                 font_size=22, color=(255, 255, 255), 
                                                 bg_color=(0, 0, 0))
    
    # Add legend (top right)
    legend_items = [(color, SPA_NAMES[cat_id]) 
                    for cat_id, color in sorted(SPA_COLORS.items()) 
                    if cat_id in SPA_NAMES]
    legend_x = img.shape[1] - 180
    overlay = draw_legend(overlay, legend_items, (legend_x, 30))
    
    cv2.imwrite(str(output_path), overlay)
    return True


def process_batch(annotation_type):
    """Process all images for a given annotation type."""
    annotation_type = annotation_type.upper()
    
    if annotation_type not in ['SPA', 'STR', 'OBJ', 'OCR']:
        print(f"Error: Invalid annotation type '{annotation_type}'")
        print("Valid types: SPA, STR, OBJ, OCR")
        return
    
    # Set paths
    image_folder = BASE_IMAGE_PATH / annotation_type
    json_folder = BASE_JSON_PATH / annotation_type
    output_folder = BASE_OUTPUT_PATH / annotation_type
    
    # Check if folders exist
    if not image_folder.exists():
        print(f"Error: Image folder not found: {image_folder}")
        return
    
    if not json_folder.exists():
        print(f"Error: JSON folder not found: {json_folder}")
        return
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Select visualization function
    viz_func = {
        'SPA': visualize_spa,
        'STR': visualize_str,
        'OBJ': visualize_obj,
        'OCR': visualize_ocr
    }[annotation_type]
    
    # Find all JSON files
    json_files = list(json_folder.glob("*.json"))
    
    if len(json_files) == 0:
        print(f"Warning: No JSON files found in {json_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {annotation_type} annotations")
    print(f"{'='*60}")
    print(f"Image folder: {image_folder}")
    print(f"JSON folder: {json_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(json_files)} JSON files")
    print(f"{'='*60}\n")
    
    # Process each file
    success_count = 0
    for i, json_path in enumerate(json_files, 1):
        # Find corresponding image
        image_name = json_path.stem + ".PNG"
        image_path = image_folder / image_name
        
        if not image_path.exists():
            image_path = image_folder / (json_path.stem + ".png")
        
        if not image_path.exists():
            print(f"[{i}/{len(json_files)}] Image not found: {json_path.name}")
            continue
        
        # Process
        output_path = output_folder / f"{json_path.stem}_viz.png"
        
        if viz_func(image_path, json_path, output_path):
            success_count += 1
            print(f"[{i}/{len(json_files)}]  {json_path.name}")
        else:
            print(f"[{i}/{len(json_files)}] Failed: {json_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Completed! Successfully processed {success_count}/{len(json_files)} files")
    print(f"Output saved to: {output_folder}")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize architectural drawing annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'type',
        type=str,
        choices=['SPA', 'STR', 'OBJ', 'OCR', 'spa', 'str', 'obj', 'ocr'],
        help='Annotation type to visualize'
    )
    
    args = parser.parse_args()
    
    process_batch(args.type)