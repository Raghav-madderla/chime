from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MLDiseasePrediction:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=3, 
                                            random_state=42, class_weight='balanced')

    def load_data(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, header=None, names=column_names)
        df.replace('?', float('nan'), inplace=True)
        df.dropna(inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        X = df.drop(columns='target')
        y = df['target'].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        print("Model accuracy:", accuracy_score(y_test, y_pred))
        print("Classification report:\n", classification_report(y_test, y_pred))
        print("\nPredicted vs Actual (first 10 examples):")
        for i in range(min(10, len(y_pred))):
            print(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")

    def run(self):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)


if __name__ == "__main__":
    predictor = MLDiseasePrediction()
    predictor.run()
