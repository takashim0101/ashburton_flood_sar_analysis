import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Define the U-Net model architecture
def unet_model(input_size=(256, 256, 2)):
    inputs = layers.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up5 = layers.concatenate([layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=-1)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=-1)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.concatenate([layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=-1)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7) # Output a single channel mask

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Function to load patches
def load_patches(data_dir):
    images = []
    masks = []
    image_files = sorted(os.listdir(os.path.join(data_dir, "images")))
    mask_files = sorted(os.listdir(os.path.join(data_dir, "masks")))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(data_dir, "images", img_file)
        mask_path = os.path.join(data_dir, "masks", mask_file)
        images.append(np.load(img_path))
        masks.append(np.load(mask_path))

    return np.array(images), np.array(masks)

def data_generator(image_generator, mask_generator):
    while True:
        img_batch = next(image_generator)
        mask_batch = next(mask_generator)
        yield img_batch, mask_batch

if __name__ == "__main__":
    # Define paths
    ml_data_dir = "ml_data"
    train_dir = os.path.join(ml_data_dir, "train")
    model_output_dir = "models"
    os.makedirs(model_output_dir, exist_ok=True)

    patch_size = 256
    input_channels = 2 # Pre-flood and post-flood SAR

    print("Loading training data patches...")
    X_train, y_train = load_patches(train_dir)

    # Ensure masks are in the correct shape for Keras (batch, height, width, channels)
    y_train = np.expand_dims(y_train, axis=-1) # Add channel dimension for masks

    print(f"Training data shape: {X_train.shape}")
    print(f"Training mask shape: {y_train.shape}")

    # Build the U-Net model
    print("Building U-Net model...")
    model = unet_model(input_size=(patch_size, patch_size, input_channels))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Data Augmentation
    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        # Add other augmentations as needed
    )
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit_generator for image and mask data to ensure the same augmentation
    seed = 1
    image_generator = image_datagen.flow(X_train, seed=seed, batch_size=4)
    mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=4)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    # Train the model with data augmentation
    print("Training U-Net model with data augmentation...")
    model.fit(
        data_generator(image_generator, mask_generator),
        steps_per_epoch=len(X_train) // 4, # batch_size
        epochs=10,
    )

    # Save the trained model
    model_save_path = os.path.join(model_output_dir, "unet_flood_detection_model.keras")
    model.save(model_save_path)
    print(f"Trained model saved to {model_save_path}")

    print("U-Net training complete.")
