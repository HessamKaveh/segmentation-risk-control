# Semantic Segmentation with FCN

Fully Convolutional Network (FCN-8s) for dense pixel-wise classification
on Pascal VOC dataset.

## Architecture
VGG16 backbone + upsampling layers with skip connections for
per-pixel class prediction.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
