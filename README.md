# Transfer Learning on Fashion MNIST — PyTorch (VGG16)

A deep learning project that applies **Transfer Learning** using a pre-trained **VGG16** model to classify Fashion MNIST images into 10 clothing categories. The model reuses powerful image features learned from ImageNet, adapting them to the Fashion MNIST task.

---

## 📌 Project Overview

This project demonstrates the power of transfer learning by:
1. Loading a **pre-trained VGG16** model trained on ImageNet
2. **Freezing the feature extraction layers** (convolutional blocks) to preserve learned features
3. **Replacing the classifier head** with a custom fully connected network for 10-class Fashion MNIST classification
4. Training only the new classifier while keeping feature weights frozen
5. Evaluating on both training and test data

---

## 🗂️ Dataset

The dataset used is the **Fashion MNIST** dataset in CSV format:
- Each row represents one image
- The first column is the **label** (class index 0–9)
- The remaining 784 columns are **pixel values** (pixel1 to pixel784), representing a 28×28 grayscale image

### Class Labels

| Label | Category       |
|-------|----------------|
| 0     | T-shirt/top    |
| 1     | Trouser        |
| 2     | Pullover       |
| 3     | Dress          |
| 4     | Coat           |
| 5     | Sandal         |
| 6     | Shirt          |
| 7     | Sneaker        |
| 8     | Bag            |
| 9     | Ankle boot     |

---

## 🧠 Model Architecture

### Transfer Learning Strategy
- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Frozen Layers:** All convolutional `features` layers — weights are not updated during training
- **Custom Classifier Head:**

```
Linear(25088 → 1024) → ReLU → Dropout(0.5)
Linear(1024 → 512)   → ReLU → Dropout(0.5)
Linear(512 → 10)
```

### Why Transfer Learning?
VGG16's convolutional layers have learned rich, general image features (edges, textures, shapes) from ImageNet. By reusing these, we benefit from powerful representations without training from scratch, saving time and data.

---

## 🔄 Data Preprocessing Pipeline

Since VGG16 was designed for 224×224 RGB images and Fashion MNIST images are 28×28 grayscale, a custom transformation pipeline converts each image:

```python
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

- Grayscale images are converted to **3-channel (RGB)** by stacking the single channel three times
- Images are resized and normalized to match ImageNet statistics

---

## ⚙️ Tech Stack

| Tool             | Purpose                              |
|------------------|--------------------------------------|
| Python 3         | Programming language                 |
| PyTorch          | Deep learning framework              |
| TorchVision      | Pre-trained models & transforms      |
| pandas           | Data loading and manipulation        |
| scikit-learn     | Train/test splitting                 |
| PIL (Pillow)     | Image processing                     |
| NumPy            | Numerical operations                 |
| matplotlib       | Visualization                        |
| CUDA (GPU)       | Accelerated training (if available)  |

---

## 🏋️ Training Configuration

| Parameter     | Value       |
|---------------|-------------|
| Optimizer     | Adam        |
| Learning Rate | 0.0001      |
| Loss Function | CrossEntropyLoss |
| Epochs        | 10          |
| Batch Size    | 32          |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/transfer-learning-fashion-mnist-pytorch.git
cd transfer-learning-fashion-mnist-pytorch
```

### 2. Install Dependencies

```bash
pip install torch torchvision pandas scikit-learn matplotlib Pillow numpy
```

> For GPU support, install the appropriate CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Add the Dataset

Download the Fashion MNIST CSV dataset (e.g., from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)) and place it in the project directory.

### 4. Run the Notebook

Open and run the notebook in Jupyter or Google Colab:

```bash
jupyter notebook transfer-learning-fashion-mnist-pytorch.ipynb
```

> **Note:** Running in Google Colab with a GPU runtime is strongly recommended, as VGG16 inference on 224×224 images is computationally intensive.

---

## 📊 Results

The model trains with a fixed random seed (`torch.manual_seed(42)`) for reproducibility. Training loss is logged per epoch, and accuracy is evaluated on both training and test sets.

> Dropout layers (p=0.5) in the classifier help combat overfitting, which is discussed and addressed in the notebook.

---

## 📁 Project Structure

```
transfer-learning-fashion-mnist-pytorch/
│
├── transfer-learning-fashion-mnist-pytorch.ipynb  # Main notebook
├── README.md                                        # Project documentation
└── data/                                            # Place Fashion MNIST CSV files here
    ├── fashion-mnist_train.csv
    └── fashion-mnist_test.csv
```

---

## 🔍 Key Concepts Covered

- **Transfer Learning** — reusing pre-trained model weights for a new task
- **Feature Extraction** — freezing convolutional layers and training only the classifier
- **Data Augmentation & Preprocessing** — adapting grayscale 28×28 images to 224×224 RGB format
- **Overfitting Solutions** — dropout regularization in the custom classifier head

---

## 🙋 Author

**Sarvagya Gupta**  
Feel free to connect or raise issues if you have suggestions or questions!

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
