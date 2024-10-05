# Handwriting Recognition
Machine learning model for recognizing **single handwritten digit**. 

Recognition is a classification problem, so model is designed for solving classification tasks. To be more specific there are totally 10 classes (one for each number 0-9).

For training **MNIST** handwritten digits dataset was used which contains 60 000 training images and 10 000 testing images.

For model testing/using a simple drawing window was made. In this window a digit can be written which can be then submit to the model. The model then returns which number is probably drawn in the input window.

## Model properties
The model takes 28x28 pixels + white digit on black background (that corresponds to training data) input image. There is 1 hidden layer that has 64 neurons with rectified linear unit (ReLU) activation function. In output layer there is softmax activation function.

For training Adaptive Moment Estimation (adam) optimizer was used and 20 training epochs.

## Dependencies
TensorFlow (keras)
- model realization
- training and testing dataset

NumPy
- work with arrays which represents images (pixels)
- input preprocessing

Tkinter
- input paint window (GUI)

Pillow
- getting input image information from tkinter
- input preprocessing

## Notes
This is my first take on machine learning problem.

Results from testing of some model parameters. **Not mentioned model parameters** are the same as described in "Model properties" chapter.

*Table 1: Testing impact of number of hidden layers + neurons on accuracy.*
*(20 training epochs)*
| Hidden layers | Neurons       | Accuracy |
|---------------|---------------|----------|
| 1             | 16            | 0.9475   |
| 1             | 32            | 0.9627   |
| 1             | 64            | 0.9735   |
| 1             | 128           | 0.9751   |
| 2             | 16 + 32       | 0.9567   |
| 2             | 32 + 64       | 0.9667   |
| 2             | 64 + 128      | 0.9685   |
| 3             | 16 + 32 + 16  | 0.9548   |
| 3             | 32 + 64 + 32  | 0.9700   |
| 3             | 64 + 128 + 64 | 0.9725   |

*Table 2: Testing impact of number of epochs on accuracy.*
*(model: 1 hidden layer with 32 neurons)*
| Epochs | Accuracy |
|--------|----------|
| 10     | 0.9608   |
| 20     | 0.9687   |
| 30     | 0.9646   |
| 40     | 0.9610   |
| 50     | 0.9609   |

There is a accuracy decreasing trend with increasing number of epochs. I assume that it is caused by overtraining this simple model. In that case the model decision becomes less general and more memorizing with training data.