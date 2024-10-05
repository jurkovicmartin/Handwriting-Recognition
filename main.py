from app import PaintWindow
from model import *

def main():
    app = PaintWindow()
    app.mainloop()

    # data = load_mnist()
    # create_model("handwritten_digits.keras", data, 20)



if __name__ == "__main__":
    main()