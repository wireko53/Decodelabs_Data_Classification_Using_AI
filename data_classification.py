from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("=== Project 2 ===")
    print("Task: Data Classification using KNN\n")

    # Load flower measurements from the built-in dataset
    iris_data = load_iris()
    flower_features = iris_data.data
    species_labels = iris_data.target

    # Reserve 20% of the data to test our model later
    train_x, test_x, train_y, test_y = train_test_split(
        flower_features, 
        species_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=species_labels
    )

    # Scale measurements so large numbers don't trick the algorithm
    data_scaler = StandardScaler()
    scaled_train_x = data_scaler.fit_transform(train_x)
    scaled_test_x = data_scaler.transform(test_x)

    # Initialize KNN with 5 neighbors and train it on our training data
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(scaled_train_x, train_y)

    # Test the model on unseen data
    model_predictions = knn_classifier.predict(scaled_test_x)

    # Print out results cleanly
    print("--- Confusion Matrix ---")
    print(confusion_matrix(test_y, model_predictions))
    
    print("\n--- Detailed Performance Metrics ---")
    print(classification_report(test_y, model_predictions, target_names=iris_data.target_names))

if __name__ == "__main__":
    main()