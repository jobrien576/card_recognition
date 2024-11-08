import unittest
from data_loader import load_data
import os

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Setup parameters for testing
        self.data_dir = r"C:\Users\jobri\OneDrive - Drexel University\MEM679\cropped_cards_data"
        self.img_height = 256
        self.img_width = 256
        self.batch_size = 32
        self.validation_split = 0.2

    def test_load_data_generators(self):
        # Test if the train and validation generators are created without error
        train_gen, val_gen = load_data(
            data_dir=self.data_dir,
            img_height=self.img_height,
            img_width=self.img_width,
            batch_size=self.batch_size,
            validation_split=self.validation_split
        )

        # Check if train_gen and val_gen are not None
        self.assertIsNotNone(train_gen, "Train generator should not be None.")
        self.assertIsNotNone(val_gen, "Validation generator should not be None.")

    def test_batch_output_shape(self):
        # Test if the batch output shape is as expected
        train_gen, _ = load_data(
            data_dir=self.data_dir,
            img_height=self.img_height,
            img_width=self.img_width,
            batch_size=self.batch_size,
            validation_split=self.validation_split
        )

        # Get a batch of images and labels
        images, labels = next(train_gen)

        # Check if the batch dimensions are correct
        self.assertEqual(images.shape, (self.batch_size, self.img_height, self.img_width, 3),
                         "Images batch shape does not match the expected dimensions.")
        self.assertEqual(labels.shape[0], self.batch_size, "Labels batch size does not match the expected batch size.")

    def test_class_mode(self):
        # Test if the class_mode is categorical
        train_gen, _ = load_data(
            data_dir=self.data_dir,
            img_height=self.img_height,
            img_width=self.img_width,
            batch_size=self.batch_size,
            validation_split=self.validation_split
        )

        # Check if the class_mode is categorical
        self.assertEqual(train_gen.class_mode, 'categorical', "Class mode should be 'categorical'.")

    def test_data_directory_exists(self):
        # Test if the data directory exists
        self.assertTrue(os.path.exists(self.data_dir), "Data directory does not exist.")

if __name__ == "__main__":
    unittest.main()
