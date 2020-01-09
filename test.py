import vklearn as vk

def test_KNeighborsClassifier():
    import sklearn.datasets
    data = sklearn.datasets.load_iris()

    from sklearn.utils import shuffle

    features = data.data
    targets = data.target

    features, targets = shuffle(features, targets)

    X_train = features[:-20]
    Y_train = targets[:-20]

    X_test = features[-20:]
    Y_test = targets[-20:]

    klf = vk.KNeighborsClassifier()
    klf.fit(X_train, Y_train)

    Y = klf.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("accuracy_score: %d" % (accuracy_score(Y, Y_test)))


if __name__ == "__main__":
    test_KNeighborsClassifier()