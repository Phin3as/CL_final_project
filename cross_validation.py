__author__ = 'Sajal'
from random import shuffle

def cross_validation(X_train, Y_train, K, model):
    X_train = list(X_train)
    shuffle(X_train)

    acc=0
    for k in range(K):
        training_x = [x for i,x in enumerate(X_train) if i % K != k]
        training_y = [y for i,y in enumerate(Y_train) if i % K != k]
        testing_x = [x for i,x in enumerate(X_train) if i % K == k]
        testing_y = [y for i,y in enumerate(Y_train) if i % K == k]
        predicted_labels = model(training_x,training_y,testing_x)
        count=0
        for i in range(len(testing_y)):
            if testing_y[i]==predicted_labels[i]:
                count+=1
        acc+=count/float(len(testing_y))
    acc = acc/float(K)

    print acc
    return acc