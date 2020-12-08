# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
from csv import reader
from math import sqrt

class KNN:
    def __init__(self):
        pass
    # Load a CSV file
    def loadCsvList(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset


    # Convert string col to float
    def colToFloat(self, dataset, col):
        for row in dataset:
            row[col] = float(row[col].strip())


    # Convert string col to integer
    def colToInt(self, dataset, col):
        class_values = [row[col] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
            print('[%s] => %d' % (value, i))
        for row in dataset:
            row[col] = lookup[row[col]]
        return lookup


    # Find the min and max values for each col
    def dataset_minmax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax


    # Rescale dataset cols to the range 0-1
    def normalData(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


    # Calculate the Euclidean distance between two vectors
    def euclidDistCalc(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)


    # Locate the most similar neighbors
    def neighborCalc(self, train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = self.euclidDistCalc(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors


    # Make a prediction with neighbors
    def predict(self, train, testRow, numNeighbors):
        neighbors = self.neighborCalc(train, testRow, numNeighbors)
        outputValues = [row[-1] for row in neighbors]
        prediction = max(set(outputValues), key=outputValues.count)
        return prediction


