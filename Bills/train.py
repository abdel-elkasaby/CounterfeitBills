import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Path to the labeled CSV file
csv_path = 'C:/Users/abdel/OneDrive/Desktop/TF Work/Bills/labels.csv'

# Read the CSV file without headers
labels_df = pd.read_csv(csv_path, header=None, names=['Filepath', 'Label'])

# Print the first few rows to ensure it's loaded correctly
print("First few rows of labels_df:")
print(labels_df.head())

# Check the shape of the DataFrame
print(f"Shape of labels_df: {labels_df.shape}")

# Check for missing values
print("Missing values in each column:")
print(labels_df.isnull().sum())

# Ensure the Filepath column contains full paths
labels_df['Filepath'] = labels_df['Filepath'].apply(lambda x: str(Path(x).resolve()))

# Convert labels to strings as required by ImageDataGenerator
labels_df['Label'] = labels_df['Label'].astype(str)

# Verify class distribution
print("Class distribution:")
print(labels_df['Label'].value_counts())

# Try splitting the dataset
try:
    train_df, test_df = train_test_split(labels_df, train_size=0.7, shuffle=True, random_state=1)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
except Exception as e:
    print(f"Error during train_test_split: {e}")

# Ensure the training and test sets are populated
if train_df.empty or test_df.empty:
    raise ValueError("The training or test DataFrame is empty after the train-test split.")

# Create ImageDataGenerators
train_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

val_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    rescale=1./255
)

# Generate batches of tensor image data
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(240, 240),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = val_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(240, 240),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(240, 240),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

# Build a model using a pre-trained model
base_model = tf.keras.applications.MobileNetV2(input_shape=(240, 240, 3), include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=1,  # Set to 1 epoch for quick testing
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=2
        )
    ]
)

# Evaluate the model on the test set
results = model.evaluate(test_images, verbose=0)
print("Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

# Make predictions with the model
raw_predictions = model.predict(test_images)
print("Raw predictions (first 10):", raw_predictions[:10])  # Print first 10 raw predictions

# Convert raw predictions to binary labels
predictions = (raw_predictions >= 0.5).astype(int).flatten()
print("Thresholded predictions (first 10):", predictions[:10])  # Print first 10 thresholded predictions

# Convert test labels to integers
test_labels = np.array(test_images.labels).astype(int)

# Generate and print the confusion matrix and classification report
cm = confusion_matrix(test_labels, predictions)
clr = classification_report(test_labels, predictions, target_names=["No Ribbon", "Ribbon"])

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=[0.5, 1.5], labels=["No Ribbon", "Ribbon"])
plt.yticks(ticks=[0.5, 1.5], labels=["No Ribbon", "Ribbon"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr)
