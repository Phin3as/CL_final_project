__author__ = 'Sajal'
from random import shuffle

def cross_validation(X_train, Y_train, K, model):
    temp = zip(X_train,Y_train)
    shuffle(temp)
    X_train = [feat for feat,label in temp]
    Y_train = [label for feat,label in temp]
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
        print 'Iteration',k+1,'Accuracy :',count/float(len(testing_y))
    acc = acc/float(K)

    print 'Final Accuracy',acc
    return acc


def cross_validation_random_partition(X_train, Y_train, K,model):
    X_train = list(X_train)
    train_prop = 0.8
    acc = 0
    for k in range(K):
        temp = zip(X_train,Y_train)
        shuffle(temp)
        X_train = [feat for feat,label in temp]
        Y_train = [label for feat,label in temp]
        train_x = X_train[0:int(train_prop*len(X_train))]
        test_x = X_train[int(train_prop*len(X_train))+1:]
        
        train_y = Y_train[0:int(train_prop*len(X_train))]
        test_y = Y_train[int(train_prop*len(X_train))+1:]

        predicted_labels = model(train_x,train_y,test_x)
        acc_i = sum([1 if predicted_labels[i]==test_y[i] else 0 for i in range(len(test_y))])
        acc+=acc_i
        print 'Iteration',k+1,'Accuracy',acc_i/float(len(test_y))
    print 'Mean Accuracy',acc/float(K)
    return acc/float(K)
