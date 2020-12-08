from csv import reader
from math import sqrt

# k-nearest neighbor learner and predictor: Elijah Flinders
class KNN:
    def __init__(self):
        pass

    # Make a prediction with neighbors
    def predict(self, train, testRow, numNeighbors):
        # set neighbors to calculated neighbors
        neighbors = self.neighborCalc(train, testRow, numNeighbors)
        outputValues = [row[-1] for row in neighbors]
        # set prediction to max values based on count
        prediction = max(set(outputValues), key=outputValues.count)
        return prediction

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
    def dataMaxMin(self, dataset):
        minmax = list()
        # loop through data and set max and min
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            # append them to minmax list
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
        # calc euclidean distance going through each component
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)


    # Calculate most similar neighbors from 
    def neighborCalc(self, train, testRow, numNeighbors):
        distances = list()
        # go through training set and grab euclid dist of each
        for trainRow in train:
            dist = self.euclidDistCalc(testRow, trainRow)
            distances.append((trainRow, dist))
        # sort the distances
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        # append neighbors based on distance
        for i in range(numNeighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # Load a CSV file
    def loadCsvListKnn(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset
