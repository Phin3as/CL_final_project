function predicted_label = predict_svm(X_train, Y_train, X_test)

mdl = fitcsvm(X_train,Y_train,'KernelFunction','kernel_intersection');

format shortg
clock

predicted_label = predict(svm_mdl,X_test);

format shortg
clock