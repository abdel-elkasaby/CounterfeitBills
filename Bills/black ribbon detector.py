from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import csv

def detect_black_stripe(image_path):
    # Load the image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Adjust the threshold value for better detection
    _, binary_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the cleaned binary image
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Define the minimum and maximum width and height for the stripe
        min_width = 5
        min_height = 30  # Adjusted to be slightly more flexible
        max_width = 100
        max_height = 400

        # Define aspect ratio and area constraints
        aspect_ratio = h / w
        min_aspect_ratio = 2
        max_aspect_ratio = 20
        min_area = 1000
        max_area = 15000
        area = w * h

        # Check if the bounding box matches the expected stripe dimensions, aspect ratio, and area
        if (min_width <= w <= max_width and min_height <= h <= max_height and
                min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
                min_area <= area <= max_area):
            # Draw the bounding box for visualization
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            return True, image

    return False, image

def initial_write_results_to_csv(image_dir, csv_path):
    image_paths = list(Path(image_dir).glob('*.jpg'))
    results = {}

    for image_path in image_paths:
        has_black_stripe, _ = detect_black_stripe(image_path)
        results[image_path] = int(has_black_stripe)

    # Write initial results to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for image_path, has_black_stripe in results.items():
            csvwriter.writerow([str(image_path), has_black_stripe])

    return results

class ImageNavigator:
    def __init__(self, image_dir, csv_path):
        self.image_dir = image_dir
        self.image_paths = list(Path(image_dir).glob('*.jpg'))
        self.current_index = 0
        self.csv_path = csv_path
        self.detection_results = initial_write_results_to_csv(image_dir, csv_path)
        self.corrections = {}

        self.root = tk.Tk()
        self.root.title("Image Navigator")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

        self.prev_button = tk.Button(self.root, text="< (Left Arrow)", command=self.prev_image)
        self.prev_button.pack(side="left")

        self.next_button = tk.Button(self.root, text="> (Right Arrow)", command=self.next_image)
        self.next_button.pack(side="right")

        self.correct_button = tk.Button(self.root, text="This has a stripe (Space)", command=self.mark_as_stripe)
        self.correct_button.pack(side="left")

        self.incorrect_button = tk.Button(self.root, text="This doesn't have a stripe (/)", command=self.mark_as_no_stripe)
        self.incorrect_button.pack(side="right")

        self.root.bind('<Left>', lambda event: self.prev_image())
        self.root.bind('<Right>', lambda event: self.next_image())
        self.root.bind('<space>', lambda event: self.mark_as_stripe())
        self.root.bind('</>', lambda event: self.mark_as_no_stripe())

        self.update_image()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def update_image(self):
        if not self.image_paths:
            self.result_label.config(text="No images found.")
            return

        image_path = self.image_paths[self.current_index]
        has_black_stripe = self.detection_results[image_path]
        _, processed_image = detect_black_stripe(image_path)

        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        im = Image.fromarray(processed_image_rgb)
        imgtk = ImageTk.PhotoImage(image=im)

        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk

        if image_path in self.corrections:
            corrected_result = self.corrections[image_path]
            if corrected_result:
                detection_text = f"Black stripe detected: ({image_path.name})"
            else:
                detection_text = f"Black stripe not detected: ({image_path.name})"
        else:
            if has_black_stripe:
                detection_text = f"Black stripe detected: ({image_path.name})"
            else:
                detection_text = f"Black stripe not detected: ({image_path.name})"
        
        self.result_label.config(text=detection_text)

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.update_image()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.update_image()

    def mark_as_stripe(self):
        image_path = self.image_paths[self.current_index]
        self.corrections[image_path] = True
        self.update_image()

    def mark_as_no_stripe(self):
        image_path = self.image_paths[self.current_index]
        self.corrections[image_path] = False
        self.update_image()

    def calculate_summary(self):
        positives = 0
        negatives = 0
        false_positives = 0
        false_negatives = 0

        for image_path in self.image_paths:
            original_detection = self.detection_results[image_path]
            corrected_detection = self.corrections.get(image_path, original_detection)

            if corrected_detection:
                if original_detection:
                    positives += 1
                else:
                    false_negatives += 1
            else:
                if original_detection:
                    false_positives += 1
                else:
                    negatives += 1

        return positives, negatives, false_positives, false_negatives

    def on_closing(self):
        positives, negatives, false_positives, false_negatives = self.calculate_summary()
        total = positives + negatives + false_positives + false_negatives
        print(f"Positives: {positives}")
        print(f"Negatives: {negatives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Total: {total}")

        # Update CSV with final results
        final_results = {**self.detection_results, **self.corrections}
        with open(self.csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for image_path, has_black_stripe in final_results.items():
                csvwriter.writerow([str(image_path), int(has_black_stripe)])

        self.root.destroy()

# Directory containing images
image_dir = 'C:/Users/abdel/OneDrive/Desktop/TF Work/Bills/Train500/'
# Path to the CSV file
csv_path = 'C:/Users/abdel/OneDrive/Desktop/TF Work/Bills/labels.csv'

# Start the image navigator
navigator = ImageNavigator(image_dir, csv_path)
