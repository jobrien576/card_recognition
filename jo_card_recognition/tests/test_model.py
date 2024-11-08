import unittest
from model import create_card_classifier, compile_model
from tensorflow.keras.models import Model

class TestModel(unittest.TestCase):

    def setUp(self):
        # Define parameters
        self.img_height = 256
        self.img_width = 256
        self.num_suits = 4
        self.num_ranks = 13

    def test_model_initialization(self):
        # Test if the model initializes correctly
        model = create_card_classifier()
        
        # Check if the model is an instance of the Keras Model class
        self.assertIsInstance(model, Model, "The model should be an instance of tensorflow.keras.models.Model.")
        
        # Check if the model has two output layers: one for suit, one for rank
        self.assertEqual(len(model.output_names), 2, "Model should have two output layers.")
        self.assertIn('suit_output', model.output_names, "Model should have an output named 'suit_output'.")
        self.assertIn('rank_output', model.output_names, "Model should have an output named 'rank_output'.")

    def test_model_compilation(self):
        # Test if the model compiles correctly with specified settings
        model = create_card_classifier()
        compiled_model = compile_model(model, learning_rate=0.001)

        # Check optimizer type
        optimizer_name = compiled_model.optimizer._name
        self.assertEqual(optimizer_name, 'Adam', "Optimizer should be 'Adam'.")

        # Check loss functions for each output
        losses = compiled_model.loss
        self.assertIn('suit_output', losses, "Model should have a loss function for 'suit_output'.")
        self.assertIn('rank_output', losses, "Model should have a loss function for 'rank_output'.")
        self.assertEqual(losses['suit_output'], 'categorical_crossentropy', 
                         "Loss for 'suit_output' should be 'categorical_crossentropy'.")
        self.assertEqual(losses['rank_output'], 'categorical_crossentropy', 
                         "Loss for 'rank_output' should be 'categorical_crossentropy'.")

    def test_model_summary(self):
        # Test if the model's summary is generated without error
        model = create_card_classifier()
        compiled_model = compile_model(model)

        # Try generating the model summary and catch any potential errors
        try:
            compiled_model.summary()
            summary_generated = True
        except Exception as e:
            summary_generated = False
            print(f"Error in generating model summary: {e}")
        
        self.assertTrue(summary_generated, "Model summary should be generated without error.")

if __name__ == "__main__":
    unittest.main()
