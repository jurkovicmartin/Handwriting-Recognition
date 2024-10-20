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

***Solution for the same problem with the same dataset is also in ML-Fundamentals repository in folder MultiLayer. In this folder is an example where i used from scratch coded model.***

Results from testing of some model parameters. **Not mentioned model parameters** are the same as described in "Model properties" chapter.

*Table 1: Testing impact of number of hidden layers + neurons on accuracy.*
*(20 training epochs)*
| Hidden layers | Neurons       | Accuracy [%] |
|---------------|---------------|--------------|
| 1             | 16            | 94.75        |
| 1             | 32            | 96.27        |
| 1             | 64            | 97.35        |
| 1             | 128           | 97.51        |
| 2             | 16 + 32       | 95.67        |
| 2             | 32 + 64       | 96.67        |
| 2             | 64 + 128      | 96.85        |
| 2             | 16 + 16       | 95.22        |
| 2             | 32 + 32       | 96.88        |
| 2             | 64 + 64       | 97.30        |
| **2**         | **128 + 128** | **97.58**    |
| 2             | 32 + 16       | 96.23        |
| 2             | 64 + 32       | 97.19        |
| 2             | 128 + 64      | 97.39        |
| 3             | 16 + 32 + 16  | 95.48        |
| 3             | 32 + 64 + 32  | 97.00        |
| 3             | 64 + 128 + 64 | 97.25        |

*Table 2: Testing impact of number of epochs on accuracy.*
*(model: 1 hidden layer with 32 neurons)*
| Epochs | Accuracy [%] |
|--------|--------------|
| 10     | 96.08        |
| 20     | 96.87        |
| 30     | 96.46        |
| 40     | 96.10        |
| 50     | 96.09        |

There is a accuracy decreasing trend with increasing number of epochs. I assume that it is caused by overtraining this simple model. In that case the model decision becomes less general and more memorizing with training data.