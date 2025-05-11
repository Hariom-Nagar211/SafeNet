
# SafeInternet - Image Safety Classification

SafeInternet is a deep learning-based image classification project aimed at detecting **unsafe or NSFW (Not Safe For Work)** content in images. It leverages a convolutional neural network (CNN) to classify images into safe and unsafe categories.

## 🧠 Features

- Classifies images as **Safe** or **Unsafe**
- Utilizes a CNN architecture built with TensorFlow/Keras
- Includes image preprocessing and data augmentation
- Visualization of training and validation performance
- Suitable for integration into content moderation pipelines

## 🗂️ Project Structure

```
SafeInternet/
├── SafeSearch_Project.ipynb     # Main notebook with model training and evaluation
├── README.md                    # Project documentation

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, Seaborn
- OpenCV (cv2)

You can install the dependencies using:

```bash
pip install tensorflow numpy matplotlib seaborn opencv-python
```

### Dataset

This project expects a dataset structured as:

```
data/
├── train/
│   ├── safe/
│   └── unsafe/
└── test/
    ├── safe/
    └── unsafe/
```

### Run the Project

Open the Jupyter Notebook:

```bash
jupyter notebook SafeSearch_Project.ipynb
```

Follow the notebook cells to:
- Load and preprocess the dataset
- Train the CNN model
- Evaluate performance with accuracy and loss plots
- Make predictions on test images

## 🧠 Model Overview

- **Architecture**: CNN with Conv2D, MaxPooling, Dropout, and Dense layers
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## 📊 Results

The notebook includes:
- Training/validation accuracy and loss plots
- Classification reports and confusion matrices
- Image-level predictions

## 📦 Future Improvements

- Convert to a real-time API using Flask or FastAPI
- Deploy the model as an Android app (using TensorFlow Lite)
- Expand dataset for better generalization
- Add multi-class classification (e.g., explicit, violence, etc.)

## 🛡️ Disclaimer

This project is for educational and research purposes only. Use responsibly and ensure proper ethical and legal considerations when deploying in real-world scenarios.
