# Blueprint Parser

Detection and OCR system for architectural floor plans (blueprints). This project benchmarks various state-of-the-art models for detecting spatial elements, objects, and structures in Korean apartment floor plans.

## Overview

The goal is to accurately detect and recognize:

### Spatial Elements (공간)
- Living room (거실), Bedroom (침실), Kitchen (주방)
- Bathroom (화장실), Entrance (현관), Balcony (발코니)
- Dressing room (드레스룸), Outdoor unit room (실외기실)
- Elevator hall (엘리베이터홀), Stairwell (계단실)
- Multi-purpose space (다목적공간), Alpha room (알파룸)

### Objects (객체)
- Toilet (변기), Sink (세면대), Kitchen sink (싱크대)
- Bathtub (욕조), Gas range (가스레인지)

### Structures (구조)
- Doors (출입문), Windows (창호), Walls (벽체)

### OCR
- Room labels, dimensions, annotations in Korean

## Dataset

The dataset is in COCO format with annotations including:
- Bounding boxes (`bbox`: [x, y, width, height])
- Category IDs mapping to element types
- OCR text in `attributes.OCR` field

### Download Sample Dataset

```bash
pip3 install gdown
gdown 1CGS27MbZCXaLpXJSjphBjvjsjO56Vtc4
```

## Models to Benchmark

End-to-end Vision-Language Models for unified detection + OCR:

| Model | Repository | Params | Notes |
|-------|------------|--------|-------|
| dots.ocr | [rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr) | 1.7B | 100+ languages, layout + OCR unified |
| Chandra | [datalab-to/chandra](https://github.com/datalab-to/chandra) | 9B | Qwen-3-VL, tables/forms/handwriting |
| DeepSeek-OCR | [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) | — | Vision-text compression, rotated text |

> **Note**: All models are open-source and support fine-tuning on custom datasets.

## Getting Started

```bash
# Clone the repository
cd blueprint-parser

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
blueprint-parser/
├── README.md
├── requirements.txt
├── data/
│   └── sample/           # Downloaded sample dataset
├── src/
│   ├── models/
│   │   ├── dots_ocr/     # dots.ocr inference & fine-tuning
│   │   ├── chandra/      # Chandra inference & fine-tuning
│   │   └── deepseek/     # DeepSeek-OCR inference & fine-tuning
│   ├── evaluation/       # Benchmarking scripts
│   └── utils/            # Data loading, preprocessing
└── notebooks/            # Experiments and analysis
```

## Evaluation Metrics

- **Layout Detection**: mAP@0.5, mAP@0.5:0.95
- **Text Recognition**: Character Error Rate (CER), Word Error Rate (WER)
- **End-to-End**: Combined accuracy (correct box + correct text)
- **Efficiency**: Inference time, GPU memory, throughput (pages/sec)

## License

TBD
