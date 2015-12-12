function predicted_label = predict_logistic(X_train, Y_train, X_test)

addpath('E:/Masters/ML/HW7/hw7_kit/hw7_kit/liblinear');

model = train(Y_train, sparse(X_train), ['-s 0', 'col']);

test_y = zeros(size(X_test,1),1);
[predicted_label] = predict(test_y, sparse(X_test), model, ['-q', 'col']);

%predicted_label = logistic(X_train, Y_train, X_test,test_y);