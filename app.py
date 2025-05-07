import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('handwritten_model.keras')


# GUI Setup
class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Buttons
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict, width=15)
        self.btn_clear = tk.Button(root, text="Clear", command=self.clear, width=15)
        self.btn_predict.grid(row=1, column=0, pady=10)
        self.btn_clear.grid(row=1, column=1, pady=10)

        # Initialize image/drawing
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Draw on canvas and PIL image
        x1, y1 = event.x - 4, event.y - 4
        x2, y2 = event.x + 4, event.y + 4
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def preprocess_image(self):
        inverted_img = ImageOps.invert(self.image)
        bbox = inverted_img.getbbox()

        if bbox:
            cropped = inverted_img.crop(bbox)
            width, height = cropped.size

            # Create a square canvas to preserve aspect ratio
            max_dim = max(width, height)
            square = Image.new("L", (max_dim, max_dim), 0)
            square.paste(cropped, ((max_dim - width) // 2, (max_dim - height) // 2))

            # Resize to 20x20 and center in 28x28
            resized = square.resize((20, 20), Image.Resampling.LANCZOS)
            final_img = Image.new("L", (28, 28), 0)
            final_img.paste(resized, (4, 4))
        else:
            final_img = Image.new("L", (28, 28), 0)

        img_array = np.array(final_img) / 255.0
        return img_array.reshape(1, 28, 28)

    def predict(self):
        processed_img = self.preprocess_image()
        logits = model.predict(processed_img)
        prediction = tf.nn.softmax(logits).numpy()
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        messagebox.showinfo("Result", f"Prediction: {digit}\nConfidence: {confidence:.2%}")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)


# Run application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()