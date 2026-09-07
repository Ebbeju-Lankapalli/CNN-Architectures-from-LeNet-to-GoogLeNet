# CNN Architectures: From LeNet-5 to GoogLeNet

A hands-on PyTorch exploration of the evolution of Convolutional Neural Networks (CNNs), from the foundational **LeNet-5** architecture to **AlexNet, VGGNet, and GoogLeNet**.

This repository implements classic CNN architectures using **PyTorch**, trains them on practical image-classification datasets, evaluates their performance, and analyzes how CNN architecture evolved over time.

---

## 📌 Overview

Convolutional Neural Networks have evolved significantly since their early applications in handwritten digit recognition.

This repository explores four important milestones in CNN history:

**LeNet-5 → AlexNet → VGGNet → GoogLeNet**

Each architecture introduces important ideas that influenced modern Computer Vision and Deep Learning.

| Architecture | Year | Dataset | Input Size | Main Contribution |
|---|---:|---|---:|---|
| **LeNet-5** | 1998 | MNIST | 32×32 | Early CNN for digit recognition |
| **AlexNet** | 2012 | ImageNette | 227×227 | Deep CNN + ReLU + Dropout |
| **VGG-16 / VGG-19** | 2014 | ImageNette | 224×224 | Deeper networks using 3×3 convolutions |
| **GoogLeNet** | 2014 | STL-10 | 96×96 | Inception modules + multi-scale feature extraction |

The goal is not only to implement these models, but also to understand **why each architectural improvement was introduced and how it affected CNN performance**.

---

# 🧠 Architectures Covered

## 1. LeNet-5

LeNet-5 is one of the earliest successful Convolutional Neural Network architectures and was originally designed for handwritten digit recognition.

### Architecture

Input: 1 × 32 × 32  
↓  
Conv2D: 1 → 6, 5×5  
↓  
Tanh  
↓  
Average Pooling  
↓  
Conv2D: 6 → 16, 5×5  
↓  
Tanh  
↓  
Average Pooling  
↓  
Flatten  
↓  
Fully Connected: 400 → 120  
↓  
Tanh  
↓  
Fully Connected: 120 → 84  
↓  
Tanh  
↓  
Fully Connected: 84 → 10

### Dataset

**MNIST**

- 60,000 training images
- 10,000 test images
- 10 digit classes
- Grayscale images
- Original size: 28×28
- Resized to 32×32

### Implementation

The LeNet-5 notebook implements the architecture using PyTorch with:

- Convolutional layers
- Tanh activation
- Average pooling
- Fully connected layers
- Cross Entropy Loss
- Adam optimizer

### Result

**Test Accuracy: 98.78%**

> Note: The implementation is LeNet-5-inspired and follows the main architectural principles of the original LeNet-5 while using modern PyTorch layers.

---

# 2. AlexNet

AlexNet was introduced in 2012 and played a major role in the deep learning revolution in Computer Vision.

It demonstrated that deeper CNNs trained with GPUs could achieve significantly better image-classification performance.

### Key Ideas

- ReLU activation
- Larger convolutional kernels
- Max pooling
- Dropout
- Local Response Normalization
- GPU-based training
- Deeper CNN architecture

### Architecture

Input: 3 × 227 × 227  
↓  
Conv2D: 3 → 96, 11×11, Stride 4  
↓  
ReLU  
↓  
Local Response Normalization  
↓  
Max Pooling  
↓  
Conv2D: 96 → 256, 5×5  
↓  
ReLU  
↓  
Local Response Normalization  
↓  
Max Pooling  
↓  
Conv2D: 256 → 384, 3×3  
↓  
ReLU  
↓  
Conv2D: 384 → 384, 3×3  
↓  
ReLU  
↓  
Conv2D: 384 → 256, 3×3  
↓  
ReLU  
↓  
Max Pooling  
↓  
Fully Connected Layers  
↓  
10 Classes

### Dataset

**ImageNette**

ImageNette is a smaller subset of ImageNet containing 10 classes.

Classes include:

- Tench
- English Springer
- Cassette Player
- Chain Saw
- Church
- French Horn
- Garbage Truck
- Gas Pump
- Golf Ball
- Parachute

### Training

The model was trained from scratch using:

- Cross Entropy Loss
- SGD optimizer
- Momentum
- Weight decay
- Step learning-rate scheduler
- Data augmentation

### Result

**Validation Accuracy: 43.03%**

The relatively low performance demonstrates the difficulty of training a large architecture like AlexNet from scratch on a relatively small dataset.

---

# 3. VGGNet

VGGNet was introduced in 2014 and showed that increasing network depth using small **3×3 convolutional filters** could significantly improve image-classification performance.

This repository implements both:

- **VGG-16**
- **VGG-19**

---

## VGG-16

### Architecture

Input: 3 × 224 × 224

Block 1  
Conv 3×3  
Conv 3×3  
Max Pool

Block 2  
Conv 3×3  
Conv 3×3  
Max Pool

Block 3  
Conv 3×3  
Conv 3×3  
Conv 3×3  
Max Pool

Block 4  
Conv 3×3  
Conv 3×3  
Conv 3×3  
Max Pool

Block 5  
Conv 3×3  
Conv 3×3  
Conv 3×3  
Max Pool

↓  
Fully Connected Layers  
↓  
10 Classes

### Parameters

Approximately:

**134.3 Million parameters**

### Result

**Validation Accuracy: 71.41%**

VGG-16 significantly improves upon the AlexNet experiment on ImageNette.

---

## VGG-19

VGG-19 follows the same basic design philosophy as VGG-16 but contains additional convolutional layers.

### Main Difference

VGG-16  
13 Convolutional Layers

↓  

VGG-19  
16 Convolutional Layers

### Parameters

Approximately:

**139.6 Million parameters**

### Result

**Validation Accuracy: ~68.6%**

Interestingly, VGG-19 performs slightly worse than VGG-16 in this experiment.

This demonstrates an important practical lesson:

> Increasing depth does not automatically guarantee better performance, especially when the dataset is relatively small.

---

# 4. GoogLeNet

GoogLeNet, introduced in 2014, introduced the **Inception architecture**.

Instead of relying on a single convolution size, Inception modules process information at multiple spatial scales simultaneously.

### Inception Module

Input

├── 1×1 Convolution  
├── 3×3 Convolution with 1×1 reduction  
├── 5×5 Convolution with 1×1 reduction  
└── Pooling Branch

↓

Concatenate

### Why Inception?

Different objects contain features at different scales.

For example:

- Small kernels capture fine details
- Medium kernels capture intermediate structures
- Larger kernels capture broader patterns

The Inception architecture allows the network to learn these features simultaneously.

---

## Auxiliary Classifiers

GoogLeNet also introduced **Auxiliary Classifiers**.

These classifiers are placed at intermediate layers of the network.

Their purpose is to:

- Improve gradient flow
- Reduce the vanishing-gradient problem
- Provide additional training signals
- Help train very deep networks

The auxiliary classifiers are used during training but are disabled during final inference.

---

## GoogLeNet Architecture

Input  
↓  
Stem Convolution Layers  
↓  
Inception 3a  
↓  
Inception 3b  
↓  
Max Pool  
↓  
Inception 4a  
↓  
Auxiliary Classifier  
↓  
Inception 4b  
↓  
Inception 4c  
↓  
Inception 4d  
↓  
Auxiliary Classifier  
↓  
Inception 4e  
↓  
Max Pool  
↓  
Inception 5a  
↓  
Inception 5b  
↓  
Global Average Pooling  
↓  
Dropout  
↓  
Fully Connected Layer  
↓  
10 Classes

### Parameters

Total parameters:

**10.34 Million**

Inference parameters:

**5.99 Million**

This is significantly fewer parameters than VGG-16 despite GoogLeNet being a deeper architecture.

---

# 📊 Experiments and Results

The following results were obtained from the experiments in this repository.

| Model | Dataset | Training Approach | Result |
|---|---|---|---:|
| **LeNet-5** | MNIST | From Scratch | **98.78% Test Accuracy** |
| **AlexNet** | ImageNette | From Scratch | **43.03% Validation Accuracy** |
| **VGG-16** | ImageNette | From Scratch | **71.41% Validation Accuracy** |
| **VGG-19** | ImageNette | From Scratch | **~68.6% Validation Accuracy** |
| **GoogLeNet** | STL-10 | From Scratch | **60.97% Test Accuracy** |
| **GoogLeNet** | STL-10 | Pretrained + Fine-tuned | **93.18% Test Accuracy** |

### ⚠️ Important

These results should **not be interpreted as a direct benchmark ranking** of the architectures because different experiments use different:

- Datasets
- Input resolutions
- Training epochs
- Training configurations
- Data splits
- Optimization strategies

The purpose of the experiments is to understand the **architectural evolution and practical behavior of CNNs**.

---

# 📈 CNN Architecture Evolution

The architectures demonstrate several important developments in Computer Vision.

### LeNet-5

Basic CNN  
↓  
Convolution  
↓  
Pooling  
↓  
Fully Connected Layers

Introduced the fundamental CNN structure.

---

### AlexNet

Deeper CNN  
+  
ReLU  
+  
Dropout  
+  
GPU Training

Demonstrated the power of deep CNNs at large scale.

---

### VGGNet

More Depth  
+  
Small 3×3 Convolutions  
+  
Simple Architecture

Showed that deeper networks with small convolutional filters can learn powerful visual representations.

---

### GoogLeNet

Deep Network  
+  
Inception Modules  
+  
Multi-scale Feature Extraction  
+  
Auxiliary Classifiers

Focused on increasing network depth and width while keeping computational and parameter costs manageable.

---

# 🔬 Key Observations

## 1. Deeper Networks Can Learn Better Representations

Moving from LeNet-5 to AlexNet and VGGNet demonstrates how increasing network depth allows models to learn increasingly complex visual features.

---

## 2. Small Convolutions Are Powerful

VGGNet showed that stacking multiple 3×3 convolutions can effectively replace larger convolutional filters while allowing the network to learn more nonlinear transformations.

---

## 3. More Parameters Do Not Always Mean Better Performance

VGG-16 and VGG-19 contain more than 130 million parameters, while GoogLeNet contains approximately 10 million parameters.

Despite having far fewer parameters, GoogLeNet is capable of learning highly effective representations.

---

## 4. Training Strategy Matters

The GoogLeNet experiment demonstrates the large difference between:

Training from Scratch  
↓  
60.97% Test Accuracy

and:

ImageNet Pretraining  
↓  
Fine-Tuning  
↓  
93.18% Test Accuracy

This highlights the importance of **transfer learning** when working with relatively small datasets.

---

## 5. More Depth Does Not Always Guarantee Better Results

VGG-19 achieved slightly lower validation accuracy than VGG-16 in this experiment.

This shows that model architecture must be considered together with:

- Dataset size
- Regularization
- Optimization
- Training duration
- Data augmentation

---

# 📂 Repository Structure

CNN-Architectures-from-LeNet-to-GoogLeNet/  
│  
├── LeNet_5_MNIST_PyTorch.ipynb  
├── AlexNet_ ImageNette.ipynb  
├── VGGNet_ ImageNette.ipynb  
├── GoogLeNet_STL10.ipynb  
├── .gitignore  
└── README.md

Each notebook contains explanations, implementation details, training, evaluation, and observations.

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

---

# 📚 Datasets

## MNIST

Used for the LeNet-5 experiment.

Contains handwritten digits from:

**0 → 9**

---

## ImageNette

Used for the AlexNet and VGGNet experiments.

ImageNette provides a smaller and more manageable subset of ImageNet for experimentation.

---

## STL-10

Used for the GoogLeNet experiment.

STL-10 contains:

- 10 classes
- 96×96 color images
- Training images
- Test images
- Unlabeled images

---

# 🚀 Getting Started

## 1. Clone the Repository

    git clone https://github.com/Ebbeju-Lankapalli/CNN-Architectures-from-LeNet-to-GoogLeNet.git

    cd CNN-Architectures-from-LeNet-to-GoogLeNet

## 2. Create a Python Environment

Using Conda:

    conda create -n cnn_env python=3.11

Activate it:

    conda activate cnn_env

## 3. Install Dependencies

    pip install torch torchvision numpy matplotlib jupyter

## 4. Launch Jupyter Notebook

    jupyter notebook

Then open any notebook and run the cells sequentially.

---

# 💻 Hardware

Training deep CNN architectures can benefit significantly from GPU acceleration.

The experiments can run on:

- CPU
- NVIDIA CUDA GPU
- Apple Silicon GPU through PyTorch MPS

For larger models such as AlexNet and VGGNet, GPU acceleration is recommended.

---

# 🎯 Learning Objectives

This repository was created to understand the evolution of CNN architectures through practical implementation.

By studying these notebooks, you can learn:

- How convolutional layers work
- How pooling reduces spatial dimensions
- How CNN architectures evolved
- How ReLU improved deep CNN training
- Why AlexNet was historically important
- Why VGG uses repeated 3×3 convolutions
- How Inception modules work
- Why GoogLeNet uses auxiliary classifiers
- How model depth affects learning
- How parameter count differs between architectures
- How transfer learning improves performance
- How to implement CNN architectures using PyTorch
- How to train and evaluate image-classification models

---

# 🧩 Architecture Comparison

| Feature | LeNet-5 | AlexNet | VGGNet | GoogLeNet |
|---|---|---|---|---|
| Main Idea | Early CNN | Deep CNN | Very Deep CNN | Inception |
| Activation | Tanh | ReLU | ReLU | ReLU |
| Convolution | 5×5 | Large + 3×3/5×5 | Mostly 3×3 | Multiple sizes |
| Pooling | Average | Max | Max | Max |
| Depth | Low | Medium | High | Very High |
| Dropout | No | Yes | Yes | Yes |
| Multi-scale Features | No | No | No | Yes |
| Auxiliary Classifiers | No | No | No | Yes |
| Transfer Learning | No | No | No | Yes |

---

# 📖 What This Project Demonstrates

This project demonstrates the transition from relatively simple CNN architectures to deeper and more computationally efficient networks.

The progression can be summarized as:

**LeNet-5 → AlexNet → VGGNet → GoogLeNet**

Each architecture solved limitations of earlier approaches and introduced ideas that influenced future CNN designs.

---

# 🔮 From GoogLeNet to Modern Computer Vision

The architectures explored here form an important part of the history of Computer Vision.

After GoogLeNet, CNN research continued toward architectures such as:

**LeNet-5 → AlexNet → VGGNet → GoogLeNet → ResNet → DenseNet → EfficientNet → Vision Transformers → Modern Vision Foundation Models**

One of the major next steps was **ResNet**, which introduced residual connections and made it practical to train extremely deep neural networks.

---

# 📌 Important Concepts Covered

### CNN Fundamentals

- Convolution
- Filters
- Feature maps
- Stride
- Padding
- Pooling
- Receptive fields

### Deep Learning

- ReLU
- Tanh
- Dropout
- Batch processing
- Cross Entropy Loss
- SGD
- Adam
- Momentum
- Learning-rate scheduling

### Computer Vision

- Image classification
- Data augmentation
- Image normalization
- Transfer learning
- Fine-tuning

### Architecture Design

- Network depth
- Parameter efficiency
- Feature extraction
- Multi-scale processing
- Auxiliary classifiers

---

# 📌 Project Highlights

- ✅ Implemented LeNet-5 using PyTorch
- ✅ Implemented AlexNet from scratch
- ✅ Implemented VGG-16 from scratch
- ✅ Implemented VGG-19 from scratch
- ✅ Implemented GoogLeNet/Inception from scratch
- ✅ Implemented Auxiliary Classifiers
- ✅ Trained models on real image datasets
- ✅ Evaluated model performance
- ✅ Compared model architectures
- ✅ Experimented with transfer learning
- ✅ Fine-tuned pretrained GoogLeNet
- ✅ Analyzed parameter counts and architecture complexity

---

# 📚 References

- LeCun et al. — *Gradient-Based Learning Applied to Document Recognition*
- Krizhevsky, Sutskever & Hinton — *ImageNet Classification with Deep Convolutional Neural Networks*
- Simonyan & Zisserman — *Very Deep Convolutional Networks for Large-Scale Image Recognition*
- Szegedy et al. — *Going Deeper with Convolutions*
- PyTorch Documentation
- Torchvision Documentation

---

# 👨‍💻 Author

**Ebbeju Lankapalli**

B.Tech Computer Science & Engineering  
Artificial Intelligence & Machine Learning

GitHub: https://github.com/Ebbeju-Lankapalli

---

# ⭐ Conclusion

This repository provides a practical exploration of the evolution of Convolutional Neural Networks.

Starting with the simple **LeNet-5**, the project progresses through **AlexNet**, **VGGNet**, and finally **GoogLeNet**, demonstrating how CNN architectures evolved to become deeper, more expressive, and more computationally efficient.

The experiments also demonstrate an important real-world machine learning principle:

> **Architecture matters, but data, optimization, regularization, and training strategy matter just as much.**

This project serves as both a learning resource and a practical implementation of several historically important CNN architectures using **PyTorch**.
