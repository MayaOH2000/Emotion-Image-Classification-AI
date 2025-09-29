# AI Image Classification Project (Academic)

## Project Overview
This academic project uses Convolutional Neural Networks (CNNs) to classify facial emotions from images. The four emotion classes are: **neutral, surprise, focused/engaged, and angry**. The project also explores **bias analysis by age** (young, middle, senior).  

The code has been modified to **remove private or sensitive data**, so you can run it with your own datasets.

---

## Dataset
- Main dataset: **FER-2013** (for most classes)  
- Folder structure:  
  - `Dataset/train`: training images for all classes  
  - `Dataset/age-bias-train`: used to test for age-related bias  
  - `Dataset/age`: additional images for bias mitigation  
- Each subfolder represents one class.  

---

## Technologies
- **Languages:** Python  
- **Libraries:** PyTorch, Skorch, NumPy, scikit-learn, Matplotlib  

---

## Installation
1. Clone the repository.  
2. Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## File Structure & Usage

### Model Files
- `cnnModel.py`: CNN architectures (CNN, CNNV1, CNNV2)  
- `train.py`: Train a model; outputs training logs, confusion matrix, and saves the best-fit model  
- `evaluate.py`: Evaluate model performance and confusion matrix; supports k-fold cross-validation  
- `load.py`: Test model on single images or custom datasets  

### Data Visualization
- `classDistribution.py`: Bar graph showing the number of images per class  
- `sampleimage.py`: Displays random images and histograms from each class  

## How to Run
1. Update `dataPath` and `modelPath` in the scripts.  
2. Select the CNN architecture by commenting/uncommenting in `train.py`.  
3. Run `train.py` to train the model.  
4. Use `evaluate.py` or `load.py` to test and analyze results.  



