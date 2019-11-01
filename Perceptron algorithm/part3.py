import numpy as np
import matplotlib.pyplot as plt
import csv


def getData(filename):
    all_data = np.loadtxt(filename, delimiter=',')
    y = np.array(all_data[:, 0])

    x = np.array(all_data[:, 1:])
    for i in range(len(y)):
        if y[i] == 3:
            y[i] = 1
        else:
            y[i] = -1
    x = np.insert(x, 0, 1.0, 1)
    return x, y


def get_kp_xixj(xi, xj, p):
    return (1 + np.dot(xi.T, xj) ** p)


def gram_matrix(x, y, p):
    # x_length=x.shape[0]
    # y_length=y.shape[0]
    # k=np.zeros((x_length,x_length))
    # for i in range(x_length):
    #     for j in range(y_length):
    #         k[i][j]=get_kp_xixj(x[i],y[j],p)
    # print(k)
    gram_mat = np.power(np.dot(x, y.T) + 1, p)
    return gram_mat


def kernel_Perception(x_train, y_train, x_val, y_val, p, iters=15):
    alpha = np.zeros((1, x_train.shape[0]))
    best_alpha = np.zeros((1, x_train.shape[0]))
    gram_train = gram_matrix(x_train, x_train, p)
    gram_val = gram_matrix(x_train, x_val, p)
    y_train = y_train.reshape(1, len(y_train))
    y_val = y_val.reshape(1, len(y_val))

    # print(y_train)
    #     #
    #     # print(gram_val.shape)
    acc_trainlist = []
    acc_vallist = []

    for iter in range(1, iters + 1):
        for n in range(0, x_train.shape[0]):
            u = 0
            for j in range(0, x_train.shape[0]):
                u += ((alpha[0][j]) * (gram_train[j][n]) * (y_train[0][j]))
            if y_train[0][n] * u <= 0:
                alpha[0][n] += 1
        wrong1 = 0

        y_train_predict = np.sign(np.dot(alpha * y_train, gram_train))

        # print(y_train_predict[0][0])
        # print(y_train[0])
        # print(wrong1)
        # print(y_train_predict)
        for i in range(len(y_train_predict[0])):
            if y_train_predict[0][i] != y_train[0][i]:
                wrong1 += 1
        train_accuracy = 1. - wrong1 / len(y_train[0])
        acc_trainlist.append(train_accuracy)

        wrong2 = 0
        y_val_predict = np.sign(np.dot(alpha * y_train, gram_val))
        for i in range(len(y_val_predict[0])):
            if y_val_predict[0][i] != y_val[0][i]:
                wrong2 += 1
        cur_acc=0
        val_accuracy = 1. - wrong2 / len(y_val[0])
        if val_accuracy>cur_acc:
            cur_acc=val_accuracy
            best_alpha=np.copy(alpha)
        acc_vallist.append(cur_acc)

        print('iteration is %d' % (iter))
        print('the accuracies for the train is %f' % (train_accuracy))
        print('the accuracies for the validation is %f' % (val_accuracy))
    best_valaccuracy = (max(acc_vallist))
    print('the best validation accuracy is %f' % (best_valaccuracy))

    return acc_trainlist, acc_vallist, best_valaccuracy, best_alpha
# (b)
def pretreatTest(filename):
    all_data=np.loadtxt(filename,delimiter=',')
    x=np.array(all_data)
    x=np.insert(x,0,1.0,1)
    return x

if __name__ == '__main__':
    ##(a)
    x_train, y_train = getData('pa2_train.csv')
    x_val, y_val = getData('pa2_valid.csv')

    p=3
    acc_train, acc_val, best_valaccuracy, best_alpha = kernel_Perception(x_train, y_train, x_val, y_val, p)
    print('best validation accuracy is',best_valaccuracy,'best alpha is',best_alpha)
    iter = np.arange(1, 16, 1)

    plt.plot(iter,acc_train,'r')
    plt.plot(iter, acc_val, 'g')
    plt.legend(['acc_train','acc_val'])
    plt.title('accuracies versus iteration number')
    plt.xlabel('iteration')
    plt.ylabel('accuracies')
    plt.show()

##(b)
    x_test=pretreatTest('pa2_test_no_label.csv')
    gram_test = gram_matrix(x_train, x_test, p)
    y_test_predict = (np.sign(np.dot(best_alpha * y_train, gram_test))).T

    np.savetxt('kplabel.csv',y_test_predict,delimiter=',')