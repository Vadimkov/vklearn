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
    print("test_KNeighborsClassifier. accuracy_score: %d" % (accuracy_score(Y, Y_test)))

def test_LinearRegressionCoordDescent():
    from sklearn import datasets
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    clnr = vk.LinearRegression(10000)
    clnr.fit(diabetes_X_train, diabetes_y_train, "coord_descent")

    Y = clnr.predict(diabetes_X_test)
    print("Y type:", type(Y), " Len:", len(Y))
    print("diabetes_y_test type:", type(diabetes_y_test), " Len:", len(diabetes_y_test))

    print("Y[0]", Y[0])
    print("diabetes_y_test[0]", diabetes_y_test[0])

    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(Y, diabetes_y_test)
    print("test_LinearRegressionCoordDescent. mean_squared_error: %d" % (error))

def test_LinearRegressionGradDescent():
    from sklearn import datasets
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    glnr = vk.LinearRegression(10000)
    glnr.fit(diabetes_X_train, diabetes_y_train, "grad_descent")

    Y = glnr.predict(diabetes_X_test)

    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(Y, diabetes_y_test)
    print("test_LinearRegressionGradDescent. mean_squared_error: %d" % (error))


def test_LinearRegressionFromSKLearn():
    from sklearn import datasets
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    from sklearn.linear_model import LinearRegression
    sklnr = LinearRegression()
    sklnr.fit(diabetes_X_train, diabetes_y_train)

    Y = sklnr.predict(diabetes_X_test)

    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(Y, diabetes_y_test)
    print("test_LinearRegressionFromSKLearn. mean_squared_error: %d" % (error))


if __name__ == "__main__":
    test_KNeighborsClassifier()
    test_LinearRegressionCoordDescent()
    test_LinearRegressionGradDescent()
    test_LinearRegressionFromSKLearn()