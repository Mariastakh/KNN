import numpy as np
import pandas as pd
import operator
import math
from sklearn import preprocessing
import collections
from scipy.spatial import distance


# def sum(dataFrame):
#     return dataFrame.sum(axis=1)

def pre_process_data(dataFrame):
    """ Pre-processing data sets: dropping '?' rows, formatting data
    and normalising values so they could be ready for algorithm implementation"""

    char_cols = dataFrame.dtypes.pipe(lambda x: x[x == 'object']).index

    for c in char_cols:
        dataFrame[c] = pd.factorize(dataFrame[c])[0]

    dataFrame = dataFrame[~(dataFrame == -1).any(axis=1)]

    dataFrame.astype('float64')
    x = dataFrame.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dataFrame = pd.DataFrame(x_scaled)
    return dataFrame

def cv_5_fold(dataFrame):
    """Cross Validation with 5-folds in adult.train.5fold.csv"""
    dataframe_collection = {}
    i = 0
    j = 0
    l = 0
    guessed_right = 0
    k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]

    k_values = []
    # array to store the accuracy evaluation for each number of K
    accuracy_values = {}

    myDict = {}
    for j in range(len(k)):  # for all values of K neighbour

        print(k[j])
        predicted_right = 0
        total_number = 0
        five_accuracies = []
        for i in range(0, 5):
            #aggregating dataframes by fold - e.g. 1 fold becomes test dataframe; 2,3,4,5 folds become one training dataframe
            trainingDataFrame = dataFrame.loc[dataFrame[15] != (i / 4.00)]
            trainingDataFrame = trainingDataFrame.drop([15], axis=1).reset_index(drop=True)
            testDataFrame = dataFrame.loc[dataFrame[15] == (i / 4.00)]
            testDataFrame = testDataFrame.drop([15], axis=1).reset_index(drop=True)

            # output is an array of predicted income values for testDataFrame
            output = knn(trainingDataFrame, testDataFrame, k[j])

            # for every fold validation loop calculate the accuracy:
            for instance in range(len(testDataFrame)):
                # checking number of right predictions
                if (output[instance] == testDataFrame[14].iloc[instance]):
                    predicted_right += 1.00
                total_number += 1.00

            # calculate accuracy as percentage of number of prediction divided by total
            accuracy = (predicted_right / total_number) * 100.0
            # add acccuracies for each of the 5 fold tests to an array
            five_accuracies.append(accuracy)

        # PROVIDE FINAL EVALUATION FOR K = J, BY FINDING OUT AVERAGE ACCURACY OF THE FIVE FOLD LOOPS:
        evaluation = 0.0
        for accuracy in range(len(five_accuracies)):
            evaluation += five_accuracies[accuracy]

        evaluation = evaluation / 5

        accuracy_values.update({k[j]: evaluation})

    accuracy_values = collections.OrderedDict(sorted(accuracy_values.items()))

    # compute which number of neigbors garners greatest accuracy:
    maxAccuracy = 0
    best_neighbour = 0
    # loop through dictionary values:
    for v in accuracy_values.items():
        # if the value is greater than the current maximum, make it the maximum
        if (v[1] > maxAccuracy):
            maxAccuracy = v[1]
            best_neighbour = v[0]

    print("Max accuracy ", maxAccuracy)
    print("Best Neighbor: ", best_neighbour)

    # make a text file containing the K-number and associated accuracy:
    str_x = "k value | accuracy" + "\n"
    for k, v in accuracy_values.items():
        str_x += str(k) + " | " + str(v) + "\n"
    print(str_x)

    text_file = open("grid.results.txt", 'w')
    text_file.write(str_x)
    text_file.close()


def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

def euclideanDistance(data1, data2):
    """Defining a function which calculates euclidean distance between two data points"""
    distance = 0
    for x in range(14):
        data1[x] = truncate(data1[x], 3)
        data2[x] = truncate(data2[x], 3)
        dist = truncate((data1[x] - data2[x]) ** 2, 3)
        distance = truncate(distance + dist, 3)

    # Final Euclidean distance between train poing and test point:
    distance = truncate(np.sqrt(distance), 3)
    return distance


def euclideanDistanceRow(testInstance, trainingSet):
    """euclidean function that will take a test instance, and a row from training data"""
    distances = {}

    for x in range(trainingSet.shape[0]):
        dist = euclideanDistance(testInstance, trainingSet.iloc[x])
        distances[x] = dist

    return distances


def knn(trainingSetData, testSetData, k):
    """Defining our KNN model"""
    trainingSet = trainingSetData.drop([14], axis=1)  # drop income
    testSet = testSetData.drop([14], axis=1)  # drop income

    distances = {}
    # this will store the distances re-sorted in ascending/descending order
    sort = {}
    # income band results (>=50k or <50K)
    incomePredictions = []

    # Calculating euclidean distance between each row of training data and test data instance
    for testInstance in range(len(testSet)):  # len(testSet)
    
        # Store current test Point:
        testInstance = testSet.iloc[testInstance]  
        
        distances = euclideanDistanceRow(testInstance, trainingSet)

        # sort the distances in order of smallest first:
        sorted_d = sorted(distances.items(), key=lambda x: x[1], reverse=False)

        neighbors = []

        # Extracting top k neighbors
        for x in range(k):
            neighbors.append(sorted_d[x])


        classVotes = {}

        # Calculating the most freq class in the neighbors
        results = {"lessThan50": 0, "moreThan50": 0}

        # creating a dataframe to which we will add the income values:

        for x in range(len(neighbors)):
            if (trainingSetData.iloc[neighbors[x][0]][14] == 0.0):
                results["lessThan50"] += 1
            elif (trainingSetData.iloc[neighbors[x][0]][14] == 1.0):
                results["moreThan50"] += 1

        print('results',results)

        if (results["lessThan50"] > results["moreThan50"]):
            incomePredictions.append(0.0)
        elif (results["lessThan50"] < results["moreThan50"]):
            incomePredictions.append(1.0)

    return incomePredictions


def confusionMatrix(testDataPredictions, testDataOriginal):
    """Confusion Matrix, performing now between test data predictions and original test data - adult.test.csv"""
    matrix = {"predicted >50K correctly as >50K": 0, "predicted >50K incorrectly as <=50K": 0,
              "predicted <=50K correctly as <=50K": 0, "predicted <=50K incorrectly as >50K": 0}

    for instance in range(len(testDataPredictions)):
        prediction = testDataPredictions[instance]
        original = testDataOriginal[14].iloc[instance]

        #calculating total number of TP,TN,FP and FN

        if prediction == 1.0 and original == 1.0:
            matrix["predicted >50K correctly as >50K"] += 1.00
        elif prediction == 0.0 and original == 1.0:
            matrix["predicted >50K incorrectly as <=50K"] += 1.00
        elif prediction == 0.0 and original == 0.0:
            matrix["predicted <=50K correctly as <=50K"] += 1.00
        elif prediction == 1.0 and original == 0.0:
            matrix["predicted <=50K incorrectly as >50K"] += 1.00

    #Making the confusion matrix look readable on console printing
    print('----------------')
    print('CONFUSION MATRIX')
    print( 'TP: ', matrix["predicted >50K correctly as >50K"], '||', 'FP: ', matrix["predicted >50K incorrectly as <=50K"])
    print('----------------')
    print('FN: ', matrix["predicted <=50K incorrectly as >50K"], '||', 'TN: ', matrix["predicted <=50K correctly as <=50K"])

    # definition of sensitivity, precision and specificity formulas
    sensitivity = matrix["predicted >50K correctly as >50K"] / (
            matrix["predicted >50K correctly as >50K"] + matrix["predicted <=50K incorrectly as >50K"])

    precision =  matrix["predicted >50K correctly as >50K"]/ (
            matrix["predicted >50K correctly as >50K"] +  matrix["predicted >50K incorrectly as <=50K"])

    specificity =  matrix["predicted <=50K correctly as <=50K"] / (
            matrix["predicted <=50K correctly as <=50K"] + matrix["predicted >50K incorrectly as <=50K"])

    print('Precision: ' + str(precision*100) + '%')
    print('Sensitivity: '+ str(sensitivity*100)+ '%')
    print('Specificity: '+ str(specificity*100) +'%')

    return matrix, precision, sensitivity, specificity


def main():

    # setting precision
    pd.set_option('precision', 9)

    # setting data labels for csv files

    DataLabels = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                  "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                  "class"]

    DataLabelsWithFold = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                          "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                          "native-country", "class", "fold"]

    # reading csv files
    df = pd.read_csv(
        "https://gist.githubusercontent.com/Joaoviana/451080b83d47fa57ee9fcdc23f9d98eb/raw/f556684c47a8de7a692fd3c0398e86ae41ce845d/adult.train.5fold.csv",
        header=None, names=DataLabelsWithFold, na_values=["?"], engine='python')
    dfTest = pd.read_csv(
        "https://gist.githubusercontent.com/Joaoviana/dd13858500083e670b0b3c9f6b088701/raw/6705061f2d85bb6dfecdf3b2fae84677db1ed723/adult.test.csv",
        header=None, names=DataLabels, sep=',\s', na_values=["?"], engine='python')


    # pre-processing data frames
    df = pre_process_data(df)
    dfTest = pre_process_data(dfTest)
    dfTrainNoFolds = df

    # this following function call below is commented; it outputs grid.results.txt the best k values for the
    # algorithm;

    #cv_5_fold(df)


    # Confusion Matrix

    # dropping folds column on training dataframe for the confusion matrix
    dfTrainNoFolds.drop([15], axis=1)  # drop folds,

    dfTestKNN = knn(dfTrainNoFolds.iloc[:2500], dfTest.iloc[:2500], 39)
    confusionMatrix(dfTestKNN, dfTest.iloc[:2500])


main()
