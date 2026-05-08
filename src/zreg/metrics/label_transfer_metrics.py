from sklearn.metrics import f1_score

def compute_f1(y_true, y_pred):
    mask = y_true != -1  # assume -1 = masked
    return f1_score(y_true[mask], y_pred[mask], average="macro")

def knn_consistency(points, labels, k=10):
    tree = KDTree(points)
    _, idx = tree.query(points, k=k+1)  # include self

    scores = []
    for i in range(len(points)):
        neighbors = idx[i][1:]  # exclude self
        same = sum(labels[j] == labels[i] for j in neighbors)
        scores.append(same / k)

    return np.mean(scores)

def label_agreement(matches, labels_A, labels_B):
    correct = 0
    for i, j, _ in matches:
        if labels_A[i] == labels_B[j]:
            correct += 1
    return correct / len(matches) if matches else 0