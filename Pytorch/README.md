# Handwriting digit recognition
 
Handwritten digit recognition with fully connected neural network. MNIST dataset was used.

This model does basically the same thing as model done with Keras TensorFlow and from scratch Backpropagation project. This time it is done with pytorch.

## Pytorch vs Keras

After using both frameworks for the first time. These are my points:

- Pytorch seems to provide more control over the model
- Keras workflow is more beginner friendly
- With Keras commands autocomplete and help didn't work well for me

## Pytorch notes

My perspective of some concepts from using pytorch for the first time. These concepts seems important or are just simply new to me.

- *Tensors* - using tensors (multiple dimension matrix)
- *Device* - specifying hardware (CPU / GPU)
- *Model* - as variable model(input) = forward pass
- *Model methods* - .train(), .eval() = doesn't change parameters
- *DataLoader* - can shuffle and split inputs into batches, iterable