import numpy as np


package_name = "vklearn"


DELTA = 0.0000000001


def euclidian_distance(v1, v2):
    return abs(sum((v1 - v2)**2)**0.5)


def get_standard_error(features, targets, weights, b):
    standard_error_sum = 0.
    for feature, target in zip(features, targets):
        standard_error_sum += ((np.sum(weights * feature) + b) - target) ** 2
    standard_error = standard_error_sum / len(features)
    return standard_error


def grad_gse(f, features, targets, weights, b):
    f_val = f(features, targets, weights, b)
    
    gradient = np.zeros((len(weights) + 1))
    w = list(weights)
    w.append(b)
    point = np.array(w)

    for i in range(len(point)):
        delta_point = list(point)
        delta_point[i] += DELTA
        delta_val = f(features, targets, delta_point[:-1], delta_point[-1])
        gradient[i] = (delta_val - f_val) / DELTA
    
    return gradient


class KNeighborsClassifier:

    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
    
    def fit(self, features, targets):
        if len(features) != len(targets):
            raise AttributeError("Features number '%d' should be equal to targets number '%d'" % (len(features), len(targets)))
        
        self.features = features
        self.targets = targets

    def predict(self, X):
        Y = list()

        for feature in X:
            distances = list()
            for (x, y) in zip(self.features, self.targets):
                distance = euclidian_distance(x, feature)
                distances.append((distance, y))
            distances.sort(key=lambda a: a[0])

            distances = distances[:self.n_neighbors]
            results = dict()
            for d in distances:
                neighbor_class = d[1]
                if neighbor_class not in results:
                    results[neighbor_class] = 0
                if d[0] == 0:
                    results[neighbor_class] = 0.000000000001
                else:
                    results[neighbor_class] += 1/d[0]
            result = max(results, key=results.get)
            Y.append(result)
        
        return Y


class LinearRegression:

    def __init__(self, max_number_of_iterations = 10000):
        self._max_number_of_iterations = max_number_of_iterations

    def fit(self, features, targets, fit_model = "grad_descent"):
        if fit_model == "coord_descent":
            self._fit_by_coord_descent(features, targets)
        elif fit_model == "grad_descent":
            self._fit_by_grad_descent(features, targets)
    
    def predict(self, X):
        Y = np.zeros(len(X))
        for i in range(len(X)):
            Y[i] = np.sum(self._weights * X[i]) + self._b
        
        return Y

    
    def _fit_by_coord_descent(self, features, targets):
        self._weights = np.full(len(features[0]), 5)
        self._b = 200

        deltas = np.full(len(self._weights), 5)
        previous_error = get_standard_error(features, targets, self._weights, self._b)
        iteration_error = previous_error

        for i in range(self._max_number_of_iterations):
            for w in range(len(self._weights)):
                self._weights[w] += deltas[w]
                current_error = get_standard_error(features, targets, self._weights, self._b)

                if current_error > previous_error:
                    deltas[w] *= -0.5
                previous_error = current_error

            current_error = get_standard_error(features, targets, self._weights, self._b)
            if current_error > previous_error:
                self._b *= -0.5
            previous_error = current_error

            if i % 100 == 0 and i != 0:               
                if (previous_error - iteration_error) < 0.00001:
                    break
                iteration_error = previous_error
        
    def _fit_by_grad_descent(self, features, targets):
        self._weights = np.full(len(features[0]), 5)
        self._b = 200

        stepsize = 0.2
        steps = np.full(len(self._weights), stepsize)
        previous_error = get_standard_error(features, targets, self._weights, self._b)
        iteration_error = previous_error

        for i in range(self._max_number_of_iterations):
            current_error = get_standard_error(features, targets, self._weights, self._b)
            g = grad_gse(get_standard_error, features, targets, self._weights, self._b)
            g *= (-1.0 * stepsize)
            self._weights = self._weights + g[:-1]
            self._b += g[-1]
                
            previous_error = current_error

            if i % 100 == 0 and i != 0:
                if abs(previous_error - iteration_error) < 0.000000001:
                    break
                iteration_error = previous_error