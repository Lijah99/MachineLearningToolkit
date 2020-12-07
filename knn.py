#Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
from csv import reader
from math import sqrt

class KNN:
    def __init__(self):
        pass
    #Load a CSV file with
    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    #Convert string column to integer
    def strColumnToInt(self, dataset, column):
        classVals = [row[column] for row in dataset]
        unique = set(classVals)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
            print('[%s] => %d' % (value, i))
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    #Convert string column to float
    def strColumnToFloat(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    #Locate the most similar neighbors
    def getNeighbors(self, train, rowTest, numNeighbors):
        distances = list()
        for rowTrain in train:
            dist = self.euclideanDistance(rowTest, rowTrain)
            distances.append((rowTrain, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(numNeighbors):
            neighbors.append(distances[i][0])
        return neighbors

    #Find the min and max values for each column
    def datasetMinMax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            colVals = [row[i] for row in dataset]
            minVal = min(colVals)
            maxVal = max(colVals)
            minmax.append([minVal, maxVal])
        return minmax

    #Calculate the Euclidean distance between two vectors
    def euclideanDistance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    #Make a prediction with neighbors
    def predict(self, train, rowTest, numNeighbors):
        neighbors = self.getNeighbors(train, rowTest, numNeighbors)
        output = [row[-1] for row in neighbors]
        prediction = max(set(output), key=output.count)
        return prediction

    #Rescale dataset columns to the range 0-1
    def normalizeDataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
