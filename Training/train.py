import json
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_config():
    scripts_path = os.path.abspath(os.path.dirname(__file__))
    json_file = os.path.join(scripts_path, "..", "config.json")
    with open(json_file, "r") as config_file:
        return json.load(config_file)

def validate_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Load config and data
config = load_config()
labels_csv = config["labels_csv"]
validate_file(labels_csv)

labels_df = pd.read_csv(labels_csv, header=None, names=['Filepath', 'Label', 'Amount'])
print(f"Loaded {len(labels_df)} rows from {labels_csv}")
labels_df['Filepath'] = labels_df['Filepath'].apply(lambda x: str(Path(x).resolve()))
labels_df['Label'] = labels_df['Label'].astype(str)

# Train-test split
train_df, test_df = train_test_split(labels_df, train_size=0.7, shuffle=True, random_state=1)

# Image generators
train_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_images = train_generator.flow_from_dataframe(
    train_df, x_col='Filepath', y_col='Label', target_size=(224, 224),
    batch_size=32, class_mode='binary', subset='training'
)

val_images = train_generator.flow_from_dataframe(
    train_df, x_col='Filepath', y_col='Label', target_size=(224, 224),
    batch_size=32, class_mode='binary', subset='validation'
)

# Model setup
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=1,
    callbacks=[early_stop]
)

# Evaluate
results = model.evaluate(val_images)
print(f"Validation Accuracy: {results[1] * 100:.2f}%")

# Predict on test set
test_generator = ImageDataGenerator(rescale=1./255)
test_images = test_generator.flow_from_dataframe(
    test_df, x_col='Filepath', y_col='Label', target_size=(224, 224),
    batch_size=32, class_mode='binary', shuffle=False
)

predictions = (model.predict(test_images) > 0.5).astype(int).flatten()
test_labels = test_images.labels

# Confusion matrix and report
cm = confusion_matrix(test_labels, predictions)
clr = classification_report(test_labels, predictions, target_names=["No Ribbon", "Ribbon"])

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n", clr)
