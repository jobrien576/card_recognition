import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import apply_augmentation  # Import the custom augmentation function

def load_data(data_dir, img_height=256, img_width=256, batch_size=32, validation_split=0.2):
    """
    Loads image data from the specified directory using Keras ImageDataGenerator with custom augmentation.
    
    Parameters:
        data_dir (str): Path to the directory containing the image folders.
        img_height (int): The height to which each image will be resized.
        img_width (int): The width to which each image will be resized.
        batch_size (int): Number of images to return in each batch.
        validation_split (float): Fraction of the data to reserve for validation.
    
    Returns:
        train_generator, val_generator: Generators for training and validation data.
    """
    # Define ImageDataGenerator with data augmentation
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        preprocessing_function=apply_augmentation,  # Apply custom augmentation
        validation_split=validation_split
    )

    # Train data generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation data generator (without augmentation)
    val_generator = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=validation_split
    ).flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    return train_generator, val_generator

# Example usage:
if __name__ == "__main__":
    data_dir = r"C:\Users\jobri\OneDrive - Drexel University\MEM679\cropped_cards_data"
    train_gen, val_gen = load_data(data_dir)
