# Handwriting Recognition

Machine learning model for recognizing single handwritten digit. 

Recognition is a classification problem, so model is designed for solving classification tasks. To be more specific there are totally 10 classes (one for each number 0-9).

For training MNIST handwritten digits dataset was used.

For model testing/using a simple drawing window was made. In this window a digit can be written which can be then submit to the model. The model then returns which number is probably drawn on in the input window.

## Model properties

The model takes 28x28 pixels + white digit on black background (that corresponds to training data) input image. There are 2 hidden layers. First one has 24 and second one 48 neurons with rectified linear unit (ReLU) activation function. In output layer there is softmax activation function.

For training Adaptive Moment Estimation (adam) optimizer was used.

## Dependencies

TensorFlow

NumPy

Tkinter

Pillow