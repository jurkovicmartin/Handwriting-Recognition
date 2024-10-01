import tkinter as tk
from PIL import ImageGrab

import tensorflow as tf
import numpy as np

class PaintWindow(tk.Tk):
    """
    Simple paint window with clear and submit button.
    
    It is for testing single digit recognition neural network.

    On submit the written digit is recognized as number.
    """
    def __init__(self):
        super().__init__()
        self.title("Simple Paint App")
        self.geometry(f"200x200+200+200")

        self.canvas = tk.Canvas(self, width=140, height=140, bg="white")
        self.canvas.pack(padx=5, pady=5)

        self.clear_btn = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.submit_btn = tk.Button(self, text="Submit", command=self.submit)
        self.submit_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        # Load the neural model
        self.model = tf.keras.models.load_model("handwritten_digits.keras")


    def draw(self, event):
        x, y = event.x, event.y
        # Thickness
        r = 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")


    def clear_canvas(self):
        self.canvas.delete("all")


    def submit(self):
        """
        Submits the canvas to the neural network and prints its prediction.
        """
        # Gets image from the canvas
        self.get_canvas_image()
        # Prepare the image for the model
        self.preprocess_image()
        # Give the image to the model and print the result
        self.predict_digit()

    
    def get_canvas_image(self):
        """
        Convert the canvas to an Image. Crops it from screenshot.
        """
        # Cropping coordinates
        left = 298
        top = 298
        right = 472
        bottom = 472
        
        self.image = ImageGrab.grab().crop((left, top, right, bottom))
    

    def preprocess_image(self):
        """
        Preprocess the image for the model.
        """
        # Resize because the model has been trained with 28x28 pixels images
        self.image = self.image.resize((28, 28))
        # Converts the image to grayscale
        self.image = self.image.convert("L")
        # Converts it to array (2D array because model expects 2D array)
        self.image = np.array([self.image])
        # Invert it because model is trained with white writing on black background
        self.image = np.invert(self.image)

    
    def predict_digit(self):
        """
        Give the image to the model and get its prediction.

        Also prints the result.
        """
        # Returns array of probabilities
        self.prediction = self.model.predict(self.image)

        # Index of highest probability
        max_index = np.argmax(self.prediction[0])

        print(f"The number is {max_index} with probability {self.prediction[0][max_index]}.")