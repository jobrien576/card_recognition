import tensorflow as tf
import numpy as np
import cv2

def custom_preprocess(img, target_size=(256, 256)):
    """
    Applies custom data augmentation on the input image, including rotation, scaling, and random shifts.
    
    Parameters:
        img (tf.Tensor): Input image tensor.
        target_size (tuple): Desired output size for the image (height, width).
    
    Returns:
        tf.Tensor: The augmented image tensor.
    """
    # Convert Tensor to NumPy array for processing
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Perform affine transformation using OpenCV
    rows, cols, ch = img.shape
    angle = np.random.uniform(-30, 30)  # Random rotation angle
    scale = np.random.uniform(0.8, 1.2)  # Random scale
    tx = np.random.uniform(-0.1, 0.1) * cols  # Random horizontal shift
    ty = np.random.uniform(-0.1, 0.1) * rows  # Random vertical shift

    # Create the transformation matrix
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    M[0, 2] += tx  # Apply the horizontal shift
    M[1, 2] += ty  # Apply the vertical shift

    # Apply the transformation
    augmented_img = cv2.warpAffine(img, M, (cols, rows))

    # Resize the image to the target size
    augmented_img = cv2.resize(augmented_img, target_size)

    # Convert back to a Tensor
    return tf.convert_to_tensor(augmented_img, dtype=tf.float32)

def random_flip(img):
    """
    Randomly flips the image horizontally or vertically.
    
    Parameters:
        img (tf.Tensor): Input image tensor.
    
    Returns:
        tf.Tensor: The randomly flipped image tensor.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img

def random_brightness(img, max_delta=0.2):
    """
    Adjusts brightness of the image by a random factor.
    
    Parameters:
        img (tf.Tensor): Input image tensor.
        max_delta (float): Maximum delta for brightness adjustment.
    
    Returns:
        tf.Tensor: The brightness-adjusted image tensor.
    """
    return tf.image.random_brightness(img, max_delta=max_delta)

def apply_augmentation(img):
    """
    Applies a sequence of data augmentation techniques to the input image.
    
    Parameters:
        img (tf.Tensor): Input image tensor.
    
    Returns:
        tf.Tensor: Augmented image tensor.
    """
    img = custom_preprocess(img)
    img = random_flip(img)
    img = random_brightness(img)
    return img
