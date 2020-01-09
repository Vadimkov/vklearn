import numpy as np

package_name = "vklearn"

class KNeighborsClassifier:

    def __init__(self, n_neighbors = 5):
        print("%s.KNeighborsClassifier created" % (package_name))

        self.n_neighbors = n_neighbors
    
    def fit(self, features, targets):
        print("fit  %d objects" % (len(features)))

        if len(features) != len(targets):
            raise AttributeError("Features number '%d' should be equal to targets number '%d'" % (len(features), len(targets)))
        
        self.features = features
        self.targets = targets
        print("Targets:", self.targets)

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
            print(distances)
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


def euclidian_distance(v1, v2):
    return abs(sum((v1 - v2)**2)**0.5)