import unittest
from penn_chime.model.ml_disease_prediction import MLDiseasePrediction
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class TestMLDiseasePrediction(unittest.TestCase):

    # Test 1: Ensure data is loaded properly and has the correct shape
    def test_data_loading(self):
        predictor = MLDiseasePrediction()
        X, y = predictor.load_data()
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        # UCI Heart Disease dataset has 13 features after processing
        self.assertEqual(X.shape[1], 13)  # Check for the correct number of features
        self.assertGreater(len(y), 0)  # Ensure target is not empty

    # Test 2: Ensure the model is trained properly
    def test_model_training(self):
        predictor = MLDiseasePrediction()
        X, y = predictor.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        predictor.train(X_train, y_train)
        self.assertIsNotNone(predictor.model)  # Ensure the model has been trained
        self.assertTrue(hasattr(predictor.model, "predict"))  # Ensure model has 'predict' method

    # Test 3: Ensure predictions are generated and have the correct length
    def test_model_prediction(self):
        predictor = MLDiseasePrediction()
        X, y = predictor.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        predictor.train(X_train, y_train)
        y_pred = predictor.predict(X_test)
        self.assertEqual(len(y_pred), len(y_test))  # Predictions should match the test set size

    # Test 4: Check if model accuracy is above 70% (or adjust based on your results)
    def test_model_accuracy_threshold(self):
        predictor = MLDiseasePrediction()
        X, y = predictor.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        predictor.train(X_train, y_train)
        y_pred = predictor.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.7)  # Check if accuracy is above 70% (adjust threshold as needed)

    # Test 5: Ensure that the model is a RandomForestClassifier
    def test_model_type(self):
        predictor = MLDiseasePrediction()
        self.assertIsInstance(predictor.model, RandomForestClassifier)  # Ensure the model is a RandomForest

    # Test 6: Ensure the model raises an error if the input shape is incorrect
    def test_prediction_shape_mismatch(self):
        predictor = MLDiseasePrediction()
        X, y = predictor.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        predictor.train(X_train, y_train)

        # Test with wrong input shape
        with self.assertRaises(ValueError):
            predictor.predict(X_test[:, :5])  # Pass fewer features than expected (less than 13 features)

    # Test 7: Ensure the model raises an error with invalid (non-numeric) data
    def test_invalid_data(self):
        predictor = MLDiseasePrediction()

        # Invalid dataset with non-numeric data
        X_invalid = [["a", "b"], ["c", "d"]]  # Non-numeric data
        y_invalid = [1, 0]

        with self.assertRaises(ValueError):
            predictor.train(X_invalid, y_invalid)

    # Test 8: Ensure the model is deterministic (consistent results with the same random state)
    def test_model_consistency(self):
        predictor = MLDiseasePrediction()
        X, y = predictor.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        predictor.train(X_train, y_train)
        y_pred_1 = predictor.predict(X_test)

        # Train and predict again
        predictor.train(X_train, y_train)
        y_pred_2 = predictor.predict(X_test)

        # The predictions should be the same if the random state is fixed
        self.assertTrue((y_pred_1 == y_pred_2).all())

if __name__ == "__main__":
    unittest.main()
