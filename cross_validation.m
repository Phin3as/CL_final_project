function predictions = cross_validation(X_train,Y_train,K,get_predict)

[N] = size(X_train,1);

ind = crossvalind('Kfold', N, K);

for i=1:K
    new_X_train = X_train((ind~=i),:);
    new_Y_train = Y_train((ind~=i),:);
    new_X_test = X_train((ind==i),:);
    new_Y_test = Y_train((ind==i),:);
    
    predict_val = get_predict(new_X_train,new_Y_train,new_X_test);
    
    acc(i) = sum(new_Y_test==predict_val);
    
    acc(i) = acc(i)/size(new_X_test,1)
    i
end

mean(acc)
    
predictions=0;