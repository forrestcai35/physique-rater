import os
import sys
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import pandas as pd
from PIL import Image

# Global variables
output_csv = "model/data/train_classic.csv"
output_dir = "model/data/images/classic/front/processed_classic_front"
input_dir = "model/data/images/classic/front/classic_front"
index_file = "model/data/images/classic/front/last_index.txt"  # File to track the last processed index
columns = [
    "image_path",
    "arms_score",
    "chest_score",
    "shoulders_score",
    "abs_score",
    "legs_score",
    "back_score",
    "definition_score"
    "proportions_score",
    "potential_score",
    "size_score",
    "vascularity_score"
]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


class DataLabelerApp(QWidget):

    def __init__(self):
        super().__init__()
        self.image_index = self.load_last_index()  # Start from the last processed index
        self.image_paths = []
        self.scores = []

        # Load images from the input directory
        self.load_images()

        self.init_ui()

    def load_images(self):
        """Load all image paths from the input directory."""
        if not os.path.exists(input_dir):
            QMessageBox.critical(self, "Error", f"Input directory does not exist: {input_dir}")
            sys.exit()

        self.image_paths = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp','.gif'))  # Include .webp
        ]

        if not self.image_paths:
            QMessageBox.critical(self, "Error", f"No images found in the input directory: {input_dir}")
            sys.exit()


    def init_ui(self):
        self.setWindowTitle("Data Labeler")
        self.setGeometry(100, 100, 800, 600)

        # Layout and Widgets
        self.layout = QVBoxLayout()

        self.start_button = QPushButton("Start Rating")
        self.start_button.clicked.connect(self.score_page)
        self.layout.addWidget(self.start_button)

        self.label = QLabel("Welcome to the Data Labeler!")
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)

    def score_page(self):
        if self.image_index < len(self.image_paths):
            # Update UI for the current image
            self.clear_layout()
            self.display_image()

            if len(self.scores) < len(columns) - 1:  # Exclude "image_path" column
                current_column = columns[len(self.scores) + 1]  # Skip "image_path"

                # Update UI for scoring
                self.label = QLabel(f"Rate: {current_column}")
                self.layout.addWidget(self.label)

                self.score_entry = QLineEdit()
                self.score_entry.returnPressed.connect(self.submit_score)
                self.layout.addWidget(self.score_entry)

                self.submit_button = QPushButton("Submit")
                self.submit_button.clicked.connect(self.submit_score)
                self.layout.addWidget(self.submit_button)

                # Automatically focus the text box for convenience
                self.score_entry.setFocus()
            else:
                self.save_score()
        else:
            self.finish_processing()

    def display_image(self):
        """Display the current image with better quality and ensure it is in JPG format."""
        image_path = self.image_paths[self.image_index]
        temp_jpg_path = os.path.splitext(image_path)[0] + ".jpg"

        # Convert to JPG format if not already
        if not image_path.lower().endswith(".jpg"):
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")  # Convert to RGB mode
                    img.save(temp_jpg_path, "JPEG")
                self.image_paths[self.image_index] = temp_jpg_path  # Update the path
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process {image_path}: {e}")
                sys.exit()

        self.label = QLabel(self)
        pixmap = QPixmap(temp_jpg_path)

        if pixmap.width() > 800 or pixmap.height() > 600:
            pixmap = pixmap.scaled(800, 600, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)

        self.label.setPixmap(pixmap)
        self.layout.addWidget(self.label)

    def submit_score(self):
        score = self.score_entry.text()
        if score:
            self.scores.append(score)
            if len(self.scores) < len(columns) - 1:
                self.score_page()
            else:
                self.save_score()
        else:
            QMessageBox.critical(self, "Error", "Please enter a valid score!")


    def save_score(self):
        new_image_name = f"{self.image_index + 1}.jpg"
        processed_path = os.path.join(output_dir, new_image_name)

        # Convert the image to JPG format and move it
        current_image_path = self.image_paths[self.image_index]
        try:
            with Image.open(current_image_path) as img:
                img = img.convert("RGB")  # Ensure RGB mode for JPG compatibility
                img.save(processed_path, "JPEG")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save {current_image_path} as JPG: {e}")
            sys.exit()

        # Save scores to CSV
        df = pd.DataFrame(
            [[processed_path] + self.scores],
            columns=columns
        )
        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
            df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv, index=False)

        # Save the current index and reset scores
        self.image_index += 1
        self.scores = []
        self.save_last_index()
        self.score_page()
        
    def finish_processing(self):
        """Handle cleanup and show completion message."""
        for image_path in self.image_paths[self.image_index:]:
            os.remove(image_path)  # Remove remaining images in the input directory

        QMessageBox.information(self, "Done", "All images have been rated!")
        sys.exit()

    def load_last_index(self):
        """Load the last processed index from the file."""
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                content = f.read().strip()
                if content.isdigit():
                    return int(content)
        return 0  # Default to 0 if the file doesn't exist or is invalid

    def save_last_index(self):
        """Save the current index to the file."""
        with open(index_file, 'w') as f:
            f.write(str(self.image_index))

    def clear_layout(self):
        """Clear all widgets from the layout."""
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataLabelerApp()
    window.show()
    sys.exit(app.exec_())


