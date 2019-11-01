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

        return dict, len(data[0])-1


def normalize(traindatadict, testdatadict, len1, len2, count_feature):
    # for key in testdatadict.keys():
    #      if key != 'dummy': #key!='price': normalize price as well
    #     #     # print(dict[key])
    #     #     max = np.max(traindatadict[key])
    #     #     min = np.min(traindatadict[key])
    #     #
    #     #     traindatadict[key] = (traindatadict[key] - min) / (max - min)
    #     #     testdatadict[key] = (testdatadict[key] - min) / (max - min)

    traindata = np.zeros((count_feature, len1))
    testdata = np.zeros((count_feature, len2))

    index = 0
    for key in testdatadict.keys():
        testdata[index] = testdatadict[key]
        traindata[index] = traindatadict[key]
        index += 1

    return  traindata, testdata


def handleDate(data):
    # data = data.astype(float);
    for i in range(0, len(data)):
        month, day, year = data[i].split('/')
        time = 365 * year + 30 * month + day
        data[i] = time


def getloss(wtx, truevals, lenth):
    loss = 0
    truevals.resize(1, lenth)
    for i in range(0, len(wtx[0])):
        loss += 0.5 * (truevals[0][i] - wtx[0][i]) * (truevals[0][i] - wtx[0][i])

    return loss


def train(learnRates):
    plt.figure(figsize=(20, 10))
    traindatadict, len1= getData("PA1_train.csv")
    testdatadict, len2 = getData("PA1_dev.csv")
    # print(len1)
    train_data, test_data = normalize(traindatadict, testdatadict, len1, len2, 21)


    randomweight = np.random.rand(20, 1)

    # print(train_data[:len(train_data)-1])

    result = {}
    for learnrate in learnRates:
        weight = randomweight
        print('')
        print('Learning rate = ' + str(learnrate))
        x = []
        trainsse_list = []
        testsse_list = []
        temp_dict = {}


        for i in range(0, 1000):
            wtx = np.dot(weight.transpose(), train_data[:len(train_data) - 1])
            trainsse = getloss(wtx, train_data[len(train_data) - 1], len(train_data[0]))
            trainsse_list.append(trainsse)

            gradient = np.dot(wtx - train_data[len(train_data) - 1],
                              train_data[:len(train_data) - 1].transpose())
            # gradient_list.append(gradient)
            normal = np.linalg.norm(gradient)
            weight = np.subtract(weight, learnrate * gradient.transpose())
            # weight_list.append(weight)

            wtx = np.dot(weight.transpose(), test_data[:len(train_data) - 1])
            testsse = getloss(wtx, test_data[len(test_data) - 1], len(test_data[0]))

            testsse_list.append(testsse)

            print('iter' + str(i) + ' Trainloss:' + '{:.3f}'.format(trainsse) + ' Testloss:' + '{:.3f}'.format(testsse))
            print('norm:' + '{:.3f}'.format(normal))

            x.append(i)
            # temp_dict[gradient] = gradient_list
            # temp_dict[weight] = weight_list
            temp_dict['weight'] = str(weight)
            temp_dict['trainSSE'] = trainsse_list
            temp_dict['testSSE'] = testsse_list

            maxfeature = np.argmax(weight)
            print(maxfeature)

            if (normal < 0.5):
                print(weight)
                break;
        result[str(learnrate)] = temp_dict
        plt.subplot(1, 2, 1)
        plt.plot(x, trainsse_list, label = str(learnrate) , linewidth = 1.5)
        plt.subplot(1, 2, 2)
        plt.plot(x, testsse_list, label = str(learnrate), linestyle='dashed',linewidth=1.5)

    with open('./data3.json', 'w') as f:
        json.dump(result, f)
    f.close()

    plt.subplot(121)
    plt.xlabel('Iteration: total of' + str(i))
    plt.ylabel('SSE on Training(No Normalizing)')
    plt.legend(loc='upper right')
    plt.title('Training loss ', fontsize=11)

    plt.subplot(122)
    plt.xlabel('Iteration: total of' + str(i))
    plt.ylabel('SSE on Testing(No Normalizing)')
    plt.legend(loc='upper right')
    plt.title('Testing loss', fontsize=11)

    plt.savefig("plot3.png")



if __name__ == '__main__':
    learnrates = [ 1e-30]
    train(learnrates)
