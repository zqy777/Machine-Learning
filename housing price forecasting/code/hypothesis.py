import csv
import numpy as np
import matplotlib.pyplot as plt
import json


def getData(path):
    with open(path) as csvfile:
        readCsv = list(csv.reader(csvfile, delimiter=','))
        data = np.transpose(readCsv)

        dict = {}

        for i in range(0, len(data)):
            dict[data[i][0]] = data[i][1:]

        handleDate(dict['date'])

        for key in dict.keys():
            dict[key] = dict[key].astype(float)
        return dict, len(data[0]) - 1


def normalize(testdatadict, length, count_feature):
    traindatadict, length1 = getData("PA1_train.csv")
    print(traindatadict)

    for key in testdatadict.keys():
        if key != 'dummy':  # key!='price': normalize price as well
            # print(dict[key])
            max = np.max(traindatadict[key])
            min = np.min(traindatadict[key])

            testdatadict[key] = (testdatadict[key] - min) / (max - min)

    testdata = np.zeros((count_feature, length))
    print(testdata.shape)
    index = 0
    for key in testdatadict.keys():
        testdata[index] = testdatadict[key]
        index += 1

    return testdata


def handleDate(data):
    # data = data.astype(float);
    for i in range(0, len(data)):
        month, day, year = data[i].split('/')
        time = 365 * year + 30 * month + day
        data[i] = time


if __name__ == '__main__':
    testdatadict, length = getData("PA1_test.csv")

    test_data = normalize(testdatadict, length, 20)
    print(test_data)

    weight = np.array([[-0.05830906],
                       [0.00452398],
                       [-0.03591693],
                       [0.14931411],
                       [-0.02379136],
                       [0.26006589],
                       [0.00312821],
                       [0.10792436],
                       [0.02215048],
                       [0.01837775],
                       [0.14528191],
                       [0.06346939],
                       [0.01884709],
                       [-0.05528384],
                       [0.00078988],
                       [-0.00835936],
                       [0.05587633],
                       [-0.0293089],
                       [0.06337444],
                       [0.0588668]])

    # print(weight.shape)

    res = np.dot(weight.transpose(), test_data)

    # print(res.shape)

    i = 0

    min = 100000000.
    with open('prediction.txt', 'w') as f:
        for i in range(0, 6000):
            price = 68.08 * res[0][i] + 0.82
            # print(price)
            if price < min:
                min = price
            f.write('{:.3f} \n'.format(price))

    f.close()

    # print(min)
