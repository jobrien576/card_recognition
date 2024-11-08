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
        model: A compiled Keras Model instance.
    """
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layers: one for suits, one for ranks
    output_suit = layers.Dense(num_suits, activation='softmax', name='suit_output')
    output_rank = layers.Dense(num_ranks, activation='softmax', name='rank_output')

    # Create model with two outputs
    model = models.Model(inputs=model.input, outputs=[output_suit, output_rank])

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
