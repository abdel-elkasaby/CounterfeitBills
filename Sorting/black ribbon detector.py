import json
import cv2
import csv
import os
import pickle
from pathlib import Path
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

# Function to load configuration
def load_config():
    """Load the configuration JSON."""
    scripts_path = os.path.abspath(os.path.dirname(__file__))
    json_file = os.path.join(scripts_path, "..", "config.json")
    with open(json_file, "r") as config_file:
        return json.load(config_file)

def validate_file(file_path):
    """Ensure the specified file exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Load configuration
config = load_config()

# Extract paths from configuration
image_dir = config["base_image_dir"]
csv_path = config["labels_csv"]
progress_file = config["progress_file"]

# Validate paths
validate_file(csv_path)  # Ensure the CSV path exists if required

# Function to detect black stripes in an image
def detect_black_stripe(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w
        area = w * h
        if 5 <= w <= 100 and 30 <= h <= 400 and 2 <= aspect_ratio <= 20 and 1000 <= area <= 15000:
            return True, image
    return False, image

# Class for interactive ribbon detection
class ImageNavigator:
    def __init__(self):
        self.image_paths = list(Path(image_dir).rglob("*.jpg"))
        self.detection_results = {}
        self.load_progress()

        self.root = Tk()
        self.root.title("Ribbon Detection")

        self.image_label = Label(self.root)
        self.image_label.pack()

        self.result_label = Label(self.root, text="")
        self.result_label.pack()

        self.progress_label = Label(self.root, text="")
        self.progress_label.pack()

        self.create_buttons()
        self.update_image()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def create_buttons(self):
        Button(self.root, text="< Previous", command=self.prev_image).pack(side="left")
        Button(self.root, text="Next >", command=self.next_image).pack(side="right")
        Button(self.root, text="Stripe", command=self.mark_as_stripe).pack(side="left")
        Button(self.root, text="No Stripe", command=self.mark_as_no_stripe).pack(side="right")

    def update_image(self):
        if not self.image_paths:
            self.result_label.config(text="No images found.")
            return

        image_path = self.image_paths[self.current_index]
        has_stripe, processed_image = detect_black_stripe(image_path)
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(processed_image_rgb))

        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk
        self.result_label.config(text=f"{'Stripe Detected' if has_stripe else 'No Stripe Detected'}: {image_path.name}")

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.update_image()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.update_image()

    def mark_as_stripe(self):
        self.detection_results[self.image_paths[self.current_index]] = True
        self.next_image()

    def mark_as_no_stripe(self):
        self.detection_results[self.image_paths[self.current_index]] = False
        self.next_image()

    def save_progress(self):
        with open(progress_file, 'wb') as file:
            pickle.dump({"current_index": self.current_index, "results": self.detection_results}, file)

    def load_progress(self):
        if Path(progress_file).exists():
            with open(progress_file, 'rb') as file:
                data = pickle.load(file)
                self.current_index = data.get("current_index", 0)
                self.detection_results = data.get("results", {})

    def on_closing(self):
        self.save_progress()
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for image_path, has_stripe in self.detection_results.items():
                csvwriter.writerow([str(image_path), int(has_stripe)])
        self.root.destroy()

if __name__ == "__main__":
    navigator = ImageNavigator()
