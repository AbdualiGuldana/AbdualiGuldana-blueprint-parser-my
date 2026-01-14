"""
python batch_convert_yolo_fixed.py \
    /workspace/AbdualiGuldana-blueprint-parser-my/Sample/02.라벨링데이터/OCR_filtered \
    /workspace/AbdualiGuldana-blueprint-parser-my/Sample/01.원천데이터/OCR \
    /workspace/Sample/yolo_data/OCR 
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def convert_single_json(json_path, images_dir, output_labels_dir, output_images_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if has data
    if not data.get('images') or not data.get('annotations'):
        return 0
    
    # Get image info
    img_info = data['images'][0]
    img_width = img_info['width']
    img_height = img_info['height']
    img_filename = img_info['file_name']
    
    # Find source image
    img_path = None
    for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
        search_name = Path(img_filename).stem + ext
        potential_paths = list(images_dir.rglob(search_name))
        if potential_paths:
            img_path = potential_paths[0]
            break
    
    if not img_path or not img_path.exists():
        print(f"⚠️  Image not found: {img_filename}")
        return 0
    
    # Convert annotations to YOLO format
    yolo_lines = []
    for ann in data['annotations']:
        class_id = ann['category_id']
        bbox = ann['bbox']
        
        # Convert to YOLO format
        x_center, y_center, width, height = coco_to_yolo_bbox(bbox, img_width, img_height)
        
        # Format: class x_center y_center width height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    if not yolo_lines:
        return 0
    
    # Save YOLO label file
    label_filename = json_path.stem + '.txt'
    label_path = output_labels_dir / label_filename
    
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    # Copy image
    dst_image = output_images_dir / img_path.name
    if not dst_image.exists():
        shutil.copy2(img_path, dst_image)
    
    return len(yolo_lines)

def batch_convert_coco(
    json_dir, 
    images_dir, 
    output_dir,
    split='train'
):

    json_dir = Path(json_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_images = output_dir / split / 'images'
    output_labels = output_dir / split / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONs
    json_files = list(json_dir.rglob('*.json'))
    
    if not json_files:
        print(f" No JSON files found in {json_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Batch Converting COCO → YOLO")
    print(f"{'='*80}")
    print(f"JSONs:  {json_dir}")
    print(f"Images: {images_dir}")
    print(f"Output: {output_dir}")
    print(f"Split:  {split}")
    print(f"Found:  {len(json_files)} JSON files")
    
    # Convert each JSON
    converted = 0
    total_annotations = 0
    skipped = 0
    
    for json_file in tqdm(json_files, desc="Converting"):
        try:
            n_anns = convert_single_json(
                json_file,
                images_dir,
                output_labels,
                output_images
            )
            
            if n_anns > 0:
                converted += 1
                total_annotations += n_anns
            else:
                skipped += 1
        
        except Exception as e:
            print(f"\n  Error: {json_file.name}: {e}")
            skipped += 1
            continue
    
    print(f"\n{'='*80}")
    print(f" Conversion Complete!")
    print(f"{'='*80}")
    print(f"Converted: {converted} images")
    print(f"Total annotations: {total_annotations}")
    print(f"Skipped: {skipped} files")
    print(f"Output: {output_dir}")
    
    # Count actual files
    img_count = len(list(output_images.glob('*.*')))
    lbl_count = len(list(output_labels.glob('*.txt')))
    
    print(f"\nVerification:")
    print(f"  Images: {img_count}")
    print(f"  Labels: {lbl_count}")
    
    # Create YAML config
    yaml_path = output_dir / 'data.yaml'
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.absolute()}
train: train/images
val: val/images

nc: 1
names: ['OCR']
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n📄 YAML config: {yaml_path}")
    
    if converted > 0:
        print(f"\n Ready to train!")
        print(f"   python -c \"from ultralytics import YOLO; YOLO('yolov12s.yaml').train(data='{yaml_path}', epochs=100, batch=16)\"")
    else:
        print(f"\n  No images converted! Check:")
        print(f"   1. Are images in {images_dir}?")
        print(f"   2. Do JSON filenames match image filenames?")
        print(f"   3. Do filtered JSONs have annotations?")

def main():
    import sys
    if len(sys.argv) < 4:
       sys.exit(1)
    json_dir = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3]
    split = sys.argv[4] if len(sys.argv) > 4 else 'train'  
    
    print(f"Split: {split} (default is 'train' if not specified)\n")
    
    batch_convert_coco(json_dir, images_dir, output_dir, split)

if __name__ == '__main__':
    main()