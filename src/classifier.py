# This project uses code from "NeuralNetworkFromScratch" (https://github.com/Bot-Academy/NeuralNetworkFromScratch) by Bot Academy.
import pathlib
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Canvas:
    def __init__(self, root, classifier):
        self.root = root
        self.classifier = classifier
        self.root.title("Classifier")

        self.canvas_size = 28
        self.scale_factor = 20
        self.canvas = tk.Canvas(
            root,
            bg='white',
            width=self.canvas_size * self.scale_factor,
            height=self.canvas_size * self.scale_factor
        )
        self.canvas.pack(pady=20)

        self.color = 'black'
        self.img = np.zeros((self.canvas_size, self.canvas_size), dtype=int)

        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, pady=20)\
        
        self.clear_button = tk.Button(
            button_frame,
            text='Clear',
            command=self.clear_canvas,
            width=10,
            height=2
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)

        self.prediction_frame = tk.Frame(root)
        self.prediction_frame.pack(side=tk.BOTTOM, pady=20)

        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas_plot = self.create_matplotlib_chart(self.prediction_frame)

        self.canvas.bind('<B1-Motion>', self.paint)

    def create_matplotlib_chart(self, parent):
        chart = FigureCanvasTkAgg(self.fig, master=parent)
        chart.draw()
        chart.get_tk_widget().pack()
        return chart

    def paint(self, event):
        x = event.x // self.scale_factor
        y = event.y // self.scale_factor

        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            self.canvas.create_rectangle(
                x * self.scale_factor,
                y * self.scale_factor,
                (x + 1) * self.scale_factor,
                (y + 1) * self.scale_factor,
                fill=self.color,
                outline=self.color
            )
            self.img[y][x] = 1
            self.classify_image()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.img = np.zeros((self.canvas_size, self.canvas_size), dtype=int)
        self.ax.clear()
        self.canvas_plot.draw()

    def classify_image(self):
        drawn_image = np.array(self.img).reshape(1, -1)
        prediction, output_probs = self.classifier.classify(drawn_image)

        self.ax.clear()
        self.ax.bar(range(10), output_probs, color='#00ffba')
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Predicted Class: {prediction}")
        self.ax.set_xlabel("Classes")
        self.ax.set_ylabel("Probability")
        self.canvas_plot.draw()

class Classifier:
    def __init__(self, use_pretrained_weights=False):
        self.learn_rate = 0.01

        if use_pretrained_weights:
            self.load_weights()
        else:
            self.load_data()
            self.initialize_weights()
            self.train()
            # Only uncomment the next line if you want to save new weights.
            # The saved weights were trained for 1000 epochs.
            #self.save_weights()

    def load_data(self):            
        current_dir = pathlib.Path(__file__).parent
        with np.load(current_dir / "mnist.npz") as f:
            self.images, self.labels = f["x_train"], f["y_train"]
        self.images = self.images.astype("float32") / 255
        self.images = np.reshape(self.images,
        (self.images.shape[0], self.images.shape[1] * self.images.shape[2])
        )
        self.labels = np.eye(10)[self.labels]

    def initialize_weights(self):
        self.input_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
        self.hidden_hidden2 = np.random.uniform(-0.5, 0.5, (20, 20))
        self.hidden2_hidden3 = np.random.uniform(-0.5, 0.5, (20, 20))
        self.hidden3_output = np.random.uniform(-0.5, 0.5, (10, 20))
        self.hidden_bias = np.zeros((20, 1))
        self.hidden2_bias = np.zeros((20, 1))
        self.hidden3_bias = np.zeros((20, 1))
        self.output_bias = np.zeros((10, 1))

    def load_weights(self):
        current_dir = pathlib.Path(__file__).parent
        weights_file = current_dir.parent / "weights" / "weights.npz"
        with np.load(weights_file) as data:
            self.input_hidden = data['input_hidden']
            self.hidden_hidden2 = data['hidden_hidden2']
            self.hidden2_hidden3 = data['hidden2_hidden3']
            self.hidden3_output = data['hidden3_output']
            self.hidden_bias = data['hidden_bias']
            self.hidden2_bias = data['hidden2_bias']
            self.hidden3_bias = data['hidden3_bias']
            self.output_bias = data['output_bias']

    def save_weights(self):
        current_dir = pathlib.Path(__file__).parent
        weights_file = current_dir.parent / "weights" / "weights.npz"
        np.savez(weights_file,
                 input_hidden=self.input_hidden,
                 hidden_hidden2=self.hidden_hidden2,
                 hidden2_hidden3=self.hidden2_hidden3,
                 hidden3_output=self.hidden3_output,
                 hidden_bias=self.hidden_bias,
                 hidden2_bias=self.hidden2_bias,
                 hidden3_bias=self.hidden3_bias,
                 output_bias=self.output_bias
        )

    def train(self):
        epochs = 20
        batch_size = 128

        for epoch in range(epochs):
            num_correct = 0
            total_loss = 0
            for start in range(0, len(self.images), batch_size):
                end = min(start + batch_size, len(self.images))
                batch_images = self.images[start:end]
                batch_labels = self.labels[start:end]

                batch_images = batch_images.T
                batch_labels = batch_labels.T

                # Forward propagation: input -> hidden
                hidden_pre = self.hidden_bias + self.input_hidden @ batch_images
                hidden = 1 / (1 + np.exp(-hidden_pre)) # Sigmoid

                # Forward propagation: hidden -> hidden2
                hidden2_pre = self.hidden2_bias + self.hidden_hidden2 @ hidden
                hidden2 = 1 / (1 + np.exp(-hidden2_pre)) # Sigmoid

                # Forward propagation: hidden2 -> hidden3
                hidden3_pre = self.hidden3_bias + self.hidden2_hidden3 @ hidden2
                hidden3 = 1 / (1 + np.exp(-hidden3_pre)) # Sigmoid

                # Forward propagation: hidden3 -> output
                output_pre = self.output_bias + self.hidden3_output @ hidden3
                exp_values = np.exp(output_pre - np.max(output_pre, axis=0)) # Softmax
                output = exp_values / np.sum(exp_values, axis=0)

                output_clipped = np.clip(output, 1e-7, 1 - 1e-7)  # Cross-entropy
                confidences = output_clipped[np.argmax(batch_labels, axis=0), np.arange(batch_labels.shape[1])]
                loss = -np.sum(np.log(confidences))
                total_loss += loss
                num_correct += np.sum(np.argmax(output, axis=0) == np.argmax(batch_labels, axis=0))

                # Backpropagation: hidden3 <- output 
                delta_o = output - batch_labels
                self.hidden3_output -= self.learn_rate * delta_o @ hidden3.T
                self.output_bias -= self.learn_rate * np.sum(delta_o, axis=1, keepdims=True)

                # Backpropagation: hidden2 <- hidden3 
                delta_h3 = (self.hidden3_output.T @ delta_o) * (hidden3 * (1 - hidden3))
                self.hidden2_hidden3 -= self.learn_rate * delta_h3 @ hidden2.T
                self.hidden3_bias -= self.learn_rate * np.sum(delta_h3, axis=1, keepdims=True)

                # Backpropagation: hidden <- hidden2 
                delta_h2 = (self.hidden2_hidden3.T @ delta_h3) * (hidden2 * (1 - hidden2))
                self.hidden_hidden2 -= self.learn_rate * delta_h2 @ hidden.T
                self.hidden2_bias -= self.learn_rate * np.sum(delta_h2, axis=1, keepdims=True)

                # Backpropagation: input <- hidden 
                delta_h = (self.hidden_hidden2.T @ delta_h2) * (hidden * (1 - hidden))
                self.input_hidden -= self.learn_rate * delta_h @ batch_images.T
                self.hidden_bias -= self.learn_rate * np.sum(delta_h, axis=1, keepdims=True)

            print(
                f"Epoch: {epoch + 1}, "
                f"Accuracy: {(num_correct / self.images.shape[0]) * 100:.2f}%, "
                f"Loss: {total_loss / (len(self.images) / batch_size):.2e}"
            )

    def classify(self, drawn_image):
        # Forward propagation: input -> hidden
        hidden_pre = self.hidden_bias + self.input_hidden @ drawn_image.T
        hidden = 1 / (1 + np.exp(-hidden_pre))

        # Forward propagation: hidden -> hidden2
        hidden2_pre = self.hidden2_bias + self.hidden_hidden2 @ hidden
        hidden2 = 1 / (1 + np.exp(-hidden2_pre))

        # Forward propagation: hidden2 -> hidden3
        hidden3_pre = self.hidden3_bias + self.hidden2_hidden3 @ hidden2
        hidden3 = 1 / (1 + np.exp(-hidden3_pre))

        # Forward propagation hidden3 -> output
        output_pre = self.output_bias + self.hidden3_output @ hidden3
        exp_values = np.exp(output_pre - np.max(output_pre))
        output = exp_values / np.sum(exp_values)

        predicted_class = np.argmax(output)
        return predicted_class, output.flatten()


def main():
    choice = input(
        "Enter 1 to use pre-trained weights or 2 to train from scratch: "
    )
    if int(choice) == 1:
        use_pretrained_weights = True
    else:
        use_pretrained_weights = False
    classifier = Classifier(use_pretrained_weights)
    root = tk.Tk()
    root.geometry("1000x1000")
    canvas = Canvas(root, classifier)
    root.mainloop()

main()