# FreqDualNet
This repository contains the implementation of the paper **"FreqDualNet: Frequency-Aware Vision Transformers for Tumor Segmentation"**.

## 1. Environment Setup
Create a new conda environment and install required libraries:

```bash
conda create -n FreqDualNet python=3.9 -y
conda activate FreqDualNet
pip install -r requirements.txt

## 2. Run the Model
To train or test the model, run:

python FreqDualNet.py

## 3. Visualization
For visualization of predictions, open the notebook and run cells:
jupyter notebook DrawingPrediction_2.ipynb
