import json
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load configuration
scripts_path = os.path.abspath(os.path.dirname(__file__))
json_file = os.path.join(scripts_path, "..", "config.json")

with open(json_file, "r") as config_file:
    config = json.load(config_file)

labels_csv = config["labels_csv"]
if not os.path.exists(labels_csv):
    raise FileNotFoundError(f"File not found: {labels_csv}")

# Load labeled data
labels_df = pd.read_csv(labels_csv, header=None, names=['Filepath', 'Label', 'Amount'])
labels_df['Filepath'] = labels_df['Filepath'].apply(lambda x: str(Path(x).resolve()))

# Ensure 'Label' and 'Amount' columns are integers
labels_df['Label'] = pd.to_numeric(labels_df['Label'], errors='coerce').fillna(0).astype(int)
labels_df['Amount'] = pd.to_numeric(labels_df['Amount'], errors='coerce').fillna(0).astype(int)

# Check for missing or invalid data
if labels_df.isnull().any().any():
    print("Found null values after conversion. Ensure all data is valid.")
    raise ValueError("Null values found in Label or Amount columns.")

# One-hot encode the 'Amount' column
unique_amounts = sorted(labels_df['Amount'].unique())  # Sort for consistent mapping
amount_mapping = {amount: idx for idx, amount in enumerate(unique_amounts)}
labels_df['Amount_Encoded'] = labels_df['Amount'].map(amount_mapping)
amount_one_hot = tf.keras.utils.to_categorical(labels_df['Amount_Encoded'], num_classes=len(unique_amounts))

# Combine labels for multi-output
labels_df['Multi_Output'] = labels_df.apply(
    lambda row: {"ribbon": row['Label'], "amount": amount_one_hot[row.name]}, axis=1
)

# Split the data
train_df, test_df = train_test_split(labels_df, train_size=0.7, shuffle=True, random_state=1)

# Data generators
def custom_data_generator(dataframe, image_column, label_column, batch_size, target_size):
    while True:
        for start in range(0, len(dataframe), batch_size):
            end = min(start + batch_size, len(dataframe))
            batch_df = dataframe[start:end]
            images = []
            ribbon_labels = []
            amount_labels = []
            for _, row in batch_df.iterrows():
                img = tf.keras.preprocessing.image.load_img(
                    row[image_column], target_size=target_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                ribbon_labels.append(row[label_column]["ribbon"])
                amount_labels.append(row[label_column]["amount"])
            yield (
                tf.convert_to_tensor(images),
                {"ribbon_output": tf.convert_to_tensor(ribbon_labels), 
                 "amount_output": tf.convert_to_tensor(amount_labels)},
            )

# Create custom data generators
train_gen = custom_data_generator(train_df, "Filepath", "Multi_Output", 32, (224, 224))
val_gen = custom_data_generator(test_df, "Filepath", "Multi_Output", 32, (224, 224))

# Define input layer
input_layer = tf.keras.Input(shape=(224, 224, 3))

# Shared base layers
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
shared_output = base_model(input_layer)
shared_output = tf.keras.layers.GlobalAveragePooling2D()(shared_output)
shared_output = tf.keras.layers.Dense(128, activation='relu')(shared_output)

# Outputs
ribbon_output = tf.keras.layers.Dense(1, activation='sigmoid', name="ribbon_output")(shared_output)
amount_output = tf.keras.layers.Dense(len(unique_amounts), activation='softmax', name="amount_output")(shared_output)

# Final model
model = tf.keras.Model(inputs=input_layer, outputs=[ribbon_output, amount_output])

# Compile the model
model.compile(
    optimizer='adam',
    loss={
        "ribbon_output": "binary_crossentropy",
        "amount_output": "categorical_crossentropy",
    },
    metrics={
        "ribbon_output": "accuracy",
        "amount_output": "accuracy",
    }
)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_df) // 32,
    validation_steps=len(test_df) // 32,
    epochs=1
)

# Evaluate the model
results = model.evaluate(val_gen, steps=len(test_df) // 32)
print(f"Ribbon Detection Accuracy: {results[3] * 100:.2f}%")
print(f"Amount Classification Accuracy: {results[4] * 100:.2f}%")

# Test set predictions
test_gen = custom_data_generator(test_df, "Filepath", "Multi_Output", 32, (224, 224))
predictions = model.predict(test_gen, steps=len(test_df) // 32)
ribbon_predictions = (predictions[0] > 0.5).astype(int).flatten()
amount_predictions = predictions[1].argmax(axis=-1)

# Ribbon classification report
ribbon_test_labels = test_df['Label'].values
print("Ribbon Classification Report:")
print(classification_report(ribbon_test_labels, ribbon_predictions))

# Amount classification report
amount_test_labels = test_df['Amount_Encoded'].values
print("Amount Classification Report:")
print(classification_report(amount_test_labels, amount_predictions))

# Confusion matrices
plt.figure(figsize=(12, 6))

# Ribbon confusion matrix
cm_ribbon = confusion_matrix(ribbon_test_labels, ribbon_predictions)
plt.subplot(1, 2, 1)
sns.heatmap(cm_ribbon, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Ribbon Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Amount confusion matrix
cm_amount = confusion_matrix(amount_test_labels, amount_predictions)
plt.subplot(1, 2, 2)
sns.heatmap(cm_amount, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Amount Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
