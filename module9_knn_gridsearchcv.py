import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_input_pairs(N):
    x_values = []
    y_values = []
    for i in range(N):
        x = float(input(f"Enter x value for pair { i + 1}: "))
        y = int(input(f"Enter y value for pair { i + 1}: "))
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values).reshape(-1, 1), np.array(y_values)

def find_best_k(train_x, train_y, test_x, test_y, k_range):
    best_k = 0
    best_accuracy = 0
    for k in k_range:
        if k > len(train_x):
            continue  # Skip k values that are larger than the training set size
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_x, train_y)
        pred_y = knn.predict(test_x)
        accuracy = accuracy_score(test_y, pred_y)
        print(f"k = {k}, Accuracy = {accuracy:.3f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    return best_k, best_accuracy

def main():
    N = int(input("Enter the number of training pairs N: "))
    print("Provide the training pairs (x, y):")
    train_x, train_y = get_input_pairs(N)

    M = int(input("Enter the number of test pairs M: "))
    print("Provide the test pairs (x, y):")
    test_x, test_y = get_input_pairs(M)

    k_range = range(1, 11)

    best_k, best_accuracy = find_best_k(train_x, train_y, test_x, test_y, k_range)

    print(f"\nBest k: {best_k}")
    print(f"Test Accuracy with best k: {best_accuracy:.3f}")

if __name__ == "__main__":
    main()
