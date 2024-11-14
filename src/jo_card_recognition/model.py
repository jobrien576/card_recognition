import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Define image dimensions and number of classes
img_height = 256
img_width = 256
num_suits = 4  # Hearts, Diamonds, Clubs, Spades
num_ranks = 13  # A, 2-10, J, Q, K

# Define the CNN model
def create_card_classifier():
    """
    Creates a Convolutional Neural Network (CNN) model for card classification.
    The model has two output layers: one for classifying suits and one for classifying ranks.
    
    Returns:
        model: A Keras Model instance.
    """
    # Define the input layer
    inputs = layers.Input(shape=(img_height, img_width, 3))

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # Fully connected layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layers: one for suits, one for ranks
    output_suit = layers.Dense(num_suits, activation='softmax', name='suit_output')(x)
    output_rank = layers.Dense(num_ranks, activation='softmax', name='rank_output')(x)

    # Create model with specified inputs and outputs
    model = models.Model(inputs=inputs, outputs=[output_suit, output_rank])

    return model

# Compile the model
def compile_model(model, learning_rate=0.001):
    """
    Compiles the CNN model with specified optimizer and loss functions for multi-output classification.
    
    Parameters:
        model: The Keras Model instance to compile.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        model: The compiled Keras Model instance.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'suit_output': 'categorical_crossentropy', 'rank_output': 'categorical_crossentropy'},
        metrics={'suit_output': 'accuracy', 'rank_output': 'accuracy'}
    )
    return model

# Example usage
if __name__ == "__main__":
    # Create the model
    model = create_card_classifier()

    # Compile the model
    model = compile_model(model)

    # Print a summary of the model
    model.summary()
