# Hand Digit Recognition Neural Network

This repository contains the implementation of a neural network model for hand digit recognition using Python and TensorFlow. The model is designed to recognize digits from images and predict the corresponding digit.

## Model Architecture

The neural network model consists of the following layers:

1. **Input Layer**: Accepts a vector of size 784 (flattened 28x28 image).
2. **Hidden Layer 1**: Fully connected layer with 25 units and ReLU activation.
3. **Hidden Layer 2**: Fully connected layer with 15 units and ReLU activation.
4. **Output Layer**: Fully connected layer with 10 units and linear activation.
5. **Softmax Layer**: Converts the output to probabilities.
6. **Argmax Layer**: Provides the final prediction by selecting the highest probability.

![Model Architecture](./images/nn3.PNG)

### Model Layers
- **Input Layer**: A vector of size 784.
- **Hidden Layer 1**: 25 units with ReLU activation.
- **Hidden Layer 2**: 15 units with ReLU activation.
- **Output Layer**: 10 units with linear activation.
- **Softmax Layer**: Converts outputs to probabilities.

## Data Preparation

The dataset consists of images of hand-written digits from the MNIST dataset. Each image is of size 28x28 pixels and is flattened into a vector of size 784.

![Data Preparation](./images/nn1.png)

### Input Data
- **Image Size**: 28x28 pixels.
- **Flattened Vector Size**: 784.

The MNIST dataset contains 60,000 training images and 10,000 test images.

![Dataset](./images/nn2.PNG)

### Dataset
- **Total Training Images**: 60,000.
- **Total Test Images**: 10,000.
- **Vector Size**: 784.

## Jupyter Notebook

The implementation of the neural network model is provided in the Jupyter Notebook `Hand_Digit_Recognition.ipynb`. This notebook includes data preprocessing, model training, and evaluation.

## Usage

To use this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-digit-recognition.git

## Model Performance

### Trainable Parameters and Accuracy Comparison

- **My Model**:
  - **Trainable Parameters**: 20,175
  - **Testing Accuracy**: 96.08%

- **Benchmark Model**:
  - **Trainable Parameters**: 1,514,187
  - **Testing Accuracy**: 99.87%

### Percentage Reduction in Trainable Parameters

The percentage reduction in trainable parameters from the benchmark model to our model is calculated as follows:

Percentage Reduction = (1 - (Our Model's Trainable Parameters / Benchmark Model's Trainable Parameters)) * 100

Percentage Reduction = (1 - (20,175 / 1,514,187)) * 100 ≈ 98.67%

### Accuracy Difference

The accuracy difference between our model and the benchmark model is calculated as follows:

Accuracy Difference = Benchmark Accuracy - Our Model's Accuracy

Accuracy Difference = 99.87% - 96.08% = 3.79%

## Conclusion

The model achieves a testing accuracy of 96.08% with a significant reduction in trainable parameters (approximately 98.67% less) compared to the benchmark model. This demonstrates the efficiency of our model in terms of computational resources while maintaining a high level of accuracy.

