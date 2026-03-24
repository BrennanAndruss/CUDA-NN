# 🧠 CUDA-NN

A custom neural network implementation in CUDA and C++. This project was my first CUDA project and served as an in-depth introduction to deep learning fundamentals.

## 📝 Features

- Supports modular network components, including linear layer, activation functions (ReLU, Sigmoid, Softmax), loss functions (MSE, CrossEntropy, fused SoftmaxCE), and SGD optimizer
- Utilizes custom optimized kernels for layers and loss functions, including block-level reductions for Softmax and shared memory tiling for matrix multiplication
- Includes data loaders for the MNIST dataset and utility functions to convert to custom Tensor objects
- Additional features and optimizations in progress...

## 🔍 What I Learned

- CUDA and C++ programming, including the use of Thrust for CUDA-compatible C++ standard library features
- Techniques for writing optimized CUDA kernels, including shared-memory tiling and block-level reductions
- Foundations of neural networks and deep learning frameworks, including the underlying mathematics and the design of frameworks for creating and training models
- Training and evaulation of neural networks, and understanding of how choices in layer types and node counts impact model effectiveness

## 📊 Results

- Trained MNIST digit classification models on the GPU with ~90% accuracy

## 🛠️ Build and Run

### Build

Prerequisites

- CMake
- CUDA

```shell
mkdir build
cd build
cmake ..
```

### Run

Data loaders for the MNIST dataset are included in ```main.cu```, and the MNIST training data can be found online. Place the training data into a ```data/``` directory and update the path if needed.

```shell
./CUDA-NN
```

## 💡 Future Work

- Features: Batching, autodiff, convolutional layers, additional layers
- Optimizations: Warp-level reductions, Tensor Core utilization