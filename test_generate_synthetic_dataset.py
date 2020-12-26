import unittest
import generate_synthetic_dataset as gsd
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class ShouldRaiseErrors(unittest.TestCase):

    def test_should_throw_error_if_n_samples_none(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_samples=None)

    def test_should_throw_error_if_n_samples_less_than_10(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_samples=5)

    def test_should_throw_error_if_n_classes_none(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_classes=None)

    def test_should_throw_error_if_n_classes_is_less_than_2(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_classes=1)

    def test_should_throw_error_if_n_features_none(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_features=None)

    def test_should_throw_error_if_n_features_is_less_than_1(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_features=0)

    def test_should_throw_error_if_class_weights_dont_add_up_to_1(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(class_weights=[0,0])

    def test_should_throw_error_if_feature_names_is_None(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(feature_names=None)

    def test_should_throw_error_if_n_features_does_not_equal_length_of_feature_names(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_features=3, feature_names=['A'])
    
    def test_should_throw_error_if_n_features_does_not_equal_length_of_feature_weights(self):
        with self.assertRaises(ValueError):
            gsd.generate_synthetic_dataset(n_features=3, feature_names=['A','B','C'], feature_weights=[0.0,0.0])

class ShouldNotRaiseErrors(unittest.TestCase):
    
    def test_should_return_dataframe(self):
        self.assertTrue(isinstance(gsd.generate_synthetic_dataset(), pd.DataFrame))

    def test_should_equal_n_samples(self):
        self.assertEqual(len(gsd.generate_synthetic_dataset(n_samples=10)),10)

    def test_should_equal_n_features_length_of_feature_names(self):
        self.assertEqual(len(gsd.generate_synthetic_dataset(n_features=2,feature_names=['A','B'], feature_weights=[0.5,0.5]).columns),3)

if __name__ == '__main__':
    unittest.main()