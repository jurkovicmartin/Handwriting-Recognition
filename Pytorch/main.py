from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from network import *

def main():
    # create_model(epochs=15, save=True)
    # return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    if not model:
        print("Model wasn't loaded. Model probably doesn't exist.")
        return
    else:
        print(model)

    ### SIMPLE MODEL USAGE

    digits_names = ["0", "1", "3", "6"]
    for name in digits_names:
        digit = Image.open(f"digits/{name}.png").convert("L")
        digit = np.array(digit)

        plt.imshow(digit, cmap="gray")
        plt.title("Digit for recognition")
        plt.show()
        # Normalizing
        digit = (255 - digit) / 255

        # Preprocess for model (Pytorch)
        digit = torch.tensor(digit, dtype=torch.float32)
        # (1, 1, 28, 28) (batch, channel, image)
        digit = digit.unsqueeze(0).unsqueeze(0)
        digit = digit.to(device)

        with torch.no_grad():
            outputs = model(digit)
            prediction = torch.argmax(outputs)
            print(f"Prediction: {prediction}")
            
            # print("Values:")
            # for i in range(10):
            #     print(f"Prediction: {i}, Values: {outputs[0][i]}")

            # Move tensor from GPU to CPU
            outputs = outputs.cpu()
            plt.bar(range(10), outputs[0])
            plt.title("Network outputs")
            plt.xlabel("Prediction")
            plt.xticks(range(10))      
            plt.ylabel("Value")
            plt.show()        




if __name__ == "__main__":
    main()