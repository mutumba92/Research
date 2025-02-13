# Malware Classification with CNNs

## Overview
This project implements a Convolutional Neural Network (CNN) model to classify malware families using the Malimg dataset. The model leverages deep learning techniques to enhance malware classification and identification, achieving high accuracy in distinguishing different malware families.

## Features
- **Dataset Used:** Malimg dataset (grayscale images of malware binaries)
- **Model Architecture:**
  - Multiple convolutional layers with ReLU activation
  - Max pooling for feature reduction
  - Fully connected dense layers
  - Softmax activation for classification
- **Evaluation Metrics:** Precision, Recall, F1-score, Confusion Matrix
- **Accuracy Achieved:** 96%
- **Libraries Used:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Scikit-learn

## Installation
To run this project, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/malware-classification.git
cd malware-classification
```
2. Train the model:
```bash
python train.py
```
3. Evaluate the model:
```bash
python evaluate.py
```

## Results
### Classification Report
```
Overall Accuracy: 96%
Macro Avg F1-Score: 0.90
Weighted Avg F1-Score: 0.96
```
### Confusion Matrix
A confusion matrix visualization is available in the results directory.

## Future Improvements
- Data Augmentation for better generalization
- Hyperparameter tuning to optimize model performance
- Federated Learning implementation for decentralized training

## License
This project is licensed under the MIT License.

## Acknowledgments
- The Malimg dataset authors
- TensorFlow/Keras community

## Contributing
Feel free to open issues or submit pull requests to improve this project!


