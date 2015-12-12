function predicted_label = predict_libsvm(X_train, Y_train, X_test)

addpath E:/Masters/ML/HW6/hw6_kit-1/hw6_kit/code/libsvm

func_i= @(X,X2) kernel_intersection(X, X2);
test_y = zeros(size(X_test,1),1);

[test_err mdl]=kernel_libsvm(X_train,Y_train,X_test,test_y,func_i);

predicted_label = mdl.yhat;

%K = func_i(X_train, X_train);
%Ktest = func_i(X_train, X_test);
%model = svmtrain(Y_train, [(1:size(K,1))' K],'-t 4 -c 0.0001');
%test_y = zeros(size(X_test,1),1);
%[yhat acc vals] = svmpredict(test_y, [(1:size(Ktest,1))' Ktest], mdl_libsvm);