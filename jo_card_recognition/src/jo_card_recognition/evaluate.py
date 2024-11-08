import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_data

# Paths and parameters
data_dir = r"C:\Users\jobri\OneDrive - Drexel University\MEM679\cropped_cards_data"
img_height = 256
img_width = 256
batch_size = 32
model_path = 'card_classifier_best_model.h5'

def load_test_data():
    """
    Loads test data using the data_loader's load_data function.
    """
    _, val_generator = load_data(
        data_dir=data_dir,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size
    )
    return val_generator

def evaluate_model(model, test_data):
    """
    Evaluates the model on test data and prints accuracy for suit and rank classifications.
    
    Parameters:
        model: The trained Keras model to evaluate.
        test_data: The test data generator.
    
    Returns:
        loss, suit_acc, rank_acc: Evaluation metrics from the model.
    """
    loss, suit_loss, rank_loss, suit_acc, rank_acc = model.evaluate(test_data)
    print(f'Suit Accuracy: {suit_acc:.4f}, Rank Accuracy: {rank_acc:.4f}')
    return loss, suit_acc, rank_acc

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss over epochs.
    
    Parameters:
        history: The history object returned from model training.
    """
    # Plot training & validation accuracy for each output
    plt.figure(figsize=(12, 6))

    # Suit accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['suit_output_accuracy'], label='Suit Train Accuracy')
    plt.plot(history.history['val_suit_output_accuracy'], label='Suit Validation Accuracy')
    plt.title('Suit Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Rank accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['rank_output_accuracy'], label='Rank Train Accuracy')
    plt.plot(history.history['val_rank_output_accuracy'], label='Rank Validation Accuracy')
    plt.title('Rank Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load test data
    test_data = load_test_data()

    # Load the best model
    model = load_model(model_path)

    # Evaluate the model
    evaluate_model(model, test_data)

    # Assuming you have saved the training history in train.py and passed it here as `history`
    # plot_training_history(history)
