import os
from data_loader import load_data
from model import create_card_classifier, compile_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths and parameters
data_dir = r"C:\Users\jobri\OneDrive - Drexel University\MEM679\cropped_cards_data"
img_height = 256
img_width = 256
batch_size = 32
epochs = 10
learning_rate = 0.001

def train_model():
    # Load the data
    train_generator, val_generator = load_data(
        data_dir=data_dir,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size
    )

    # Create and compile the model
    model = create_card_classifier()
    model = compile_model(model, learning_rate=learning_rate)

    # Callbacks
    checkpoint = ModelCheckpoint(
        'card_classifier_best_model.keras', monitor='val_loss', save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, verbose=1, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping]
    )

    return model, history

# Example usage
if __name__ == "__main__":
    model, history = train_model()


# Save the model to a file
model.save("card_classifier.h5")
