function predicted_label = predict_d_tree(X_train, Y_train, X_test)

mdl = fitctree(X_train,Y_train);
predicted_label = predict(mdl,X_test);