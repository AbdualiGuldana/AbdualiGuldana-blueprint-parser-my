"""
Run:
    input_json: Path to input JSON
    output_json: Path to output filtered JSON
    task: 'OBJ', 'STR', 'SPA', or 'OCR'
    for example:
    python filter_coco.py /workspace/AbdualiGuldana-blueprint-parser-my/Sample/02.라벨링데이터/OCR /workspace/AbdualiGuldana-blueprint-parser-my/Sample/02.라벨링데이터/OCR_filtered OCR
"""

import json
from pathlib import Path
from tqdm import tqdm

task_categories = {
    'OBJ': {4: 0, 5: 1, 6: 2, 7: 3, 8: 4},
    'STR': {9: 0, 10: 1, 11: 2},
    'SPA': {1: 0, 2: 1, 3: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, 22: 11, 23: 12},
    'OCR': {21: 0},
}

class_names = {
    "OBJ": ["객체_변기", "객체_세면대", "객체_싱크대", "객체_욕조", "객체_가스레인지"],
    "STR": ["구조_출입문", "구조_창호", "구조_벽체"],
    "SPA": [
        "공간_다목적공간", "공간_엘리베이터홀", "공간_계단실", "공간_거실",
        "공간_침실", "공간_주방", "공간_현관", "공간_발코니", "공간_화장실",
        "공간_실외기실", "공간_드레스룸", "공간_기타", "공간_엘리베이터",
    ],
    "OCR": ["OCR"],
}

def filter_json(input_json, output_json, task='OBJ'):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get category mapping for this task
    cat_mapping = task_categories[task]
    
    # Filter and remap categories
    filtered_categories = []
    for new_id, name in enumerate(class_names[task]):
        filtered_categories.append({
            'id': new_id,
            'name': name
        })
    
    # Filter annotations
    filtered_annotations = []
    for ann in data['annotations']:
        cat_id = ann['category_id']
        
        # FILTER: Only keep categories for this task
        if cat_id not in cat_mapping:
            continue
        
        # REMAP to 0-indexed
        new_ann = ann.copy()
        new_ann['category_id'] = cat_mapping[cat_id]
        filtered_annotations.append(new_ann)
    
    # Create filtered JSON
    filtered_data = {
        'categories': filtered_categories,
        'images': data['images'],
        'annotations': filtered_annotations
    }
    
    # Save
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    return len(filtered_annotations)

def filter_directory(input_dir, output_dir, task='OBJ'):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all JSON files
    json_files = list(input_dir.rglob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Filtering {task} JSONs")
    print(f"{'='*80}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found:  {len(json_files)} JSON files")
    print(f"\nFiltering categories:")
    
    cat_mapping = task_categories[task]
    for old_id, new_id in cat_mapping.items():
        print(f"  {old_id} → {new_id}")
    
    # Filter each JSON
    total_annotations = 0
    filtered_count = 0
    skipped_count = 0
    
    for json_file in tqdm(json_files, desc="Filtering"):
        # Preserve directory structure
        rel_path = json_file.relative_to(input_dir)
        output_json = output_dir / rel_path
        
        try:
            n_anns = filter_json(json_file, output_json, task)
            
            if n_anns > 0:
                filtered_count += 1
                total_annotations += n_anns
            else:
                skipped_count += 1
                # Delete empty JSON
                output_json.unlink()
        
        except Exception as e:
            print(f"\n⚠️  Error processing {json_file}: {e}")
            skipped_count += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"Filtering Complete!")
    print(f"{'='*80}")
    print(f"Filtered: {filtered_count} files")
    print(f"Total annotations: {total_annotations}")
    print(f"Skipped: {skipped_count} files (no {task} annotations)")
    print(f"Output: {output_dir}")
    print(f"\n💡 Next step:")
    print(f"  Use convert_coco() on the filtered JSONs in {output_dir}")

def main():
    import sys
    
    if len(sys.argv) < 4:
        print("enter right run command")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    task = sys.argv[3].upper()
    
    if task not in task_categories:
        print(f"Invalid task: {task}")
        print(f"Valid tasks: {list(task_categories.keys())}")
        sys.exit(1)
    
    filter_directory(input_dir, output_dir, task)

if __name__ == '__main__':
    main()