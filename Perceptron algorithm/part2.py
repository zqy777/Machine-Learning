import numpy as np
import matplotlib.pyplot as plt
import csv

def getData(filename):
    all_data=np.loadtxt(filename, delimiter=',')
    y=np.array(all_data[:,0])

    x=np.array(all_data[:,1:])
    for i in range(len(y)):
        if y[i]==3:
            y[i]=1
        else:
            y[i]=-1
    x=np.insert(x,0,1.0,1)
    return x,y

def pretreatTest(filename):
    all_data=np.loadtxt(filename,delimiter=',')
    x=np.array(all_data)
    x=np.insert(x,0,1.0,1)
    return x

def average_Perception(x_train,y_train,x_val,y_val,iters=15):
    w=np.zeros(x_train.shape[1])
    w_ave=np.zeros(x_train.shape[1])
    s=1
    acc_train=[]
    acc_val=[]
    wlist=[]
    w_avelist=[]
    for iter in range(1,iters+1):
        for n in range(x_train.shape[0]):

            if np.sign(y_train[n] * (np.dot(x_train[n],w.T))) <=0:
                w=w+y_train[n]*x_train[n]
            w_ave=((s*w_ave)+w)/(s+1)
            s+=1
        train_accuracy=get_accurancy(x_train,y_train,w_ave)
        acc_train.append(train_accuracy)
        val_accuracy=get_accurancy(x_val,y_val,w_ave)
        acc_val.append(val_accuracy)
        wlist.append(w)
        w_avelist.append(w_ave)
        print('iteration is %d' %(iter))
        print('the accuracies for the train is %f' %(train_accuracy))
        print('the accuracies for the validation is %f' % (val_accuracy))
    return acc_train, acc_val, wlist, iters, w_avelist
def get_accurancy(x,y,w):
    all,wrong=0,0
    for i in range(x.shape[0]):
        if np.sign(np.dot(x[i],w.T)) != y[i]:
            wrong+=1
        all+=1
    return (all-wrong)/all

if __name__=='__main__':
    ##(a)
    x_train, y_train = getData('pa2_train.csv')
    x_val, y_val =getData('pa2_valid.csv')
    acc_train, acc_val, wlist,iters, w_avelist=average_Perception(x_train,y_train,x_val,y_val)
    iter=np.arange(1,16,1)
    plt.plot(iter,acc_train,'r')
    plt.plot(iter, acc_val, 'g')
    plt.legend(['acc_train','acc_val'])
    plt.title('accuracies versus iteration number')
    plt.xlabel('iteration')
    plt.ylabel('accuracies')
    plt.show()
    ##(c)

    x_test=pretreatTest('pa2_test_no_label.csv')
    w_test=wlist[14]
    y_text=np.sign(np.dot(x_test,(w_test.T)))
    np.savetxt('aplabel.csv',y_text,delimiter=',')
