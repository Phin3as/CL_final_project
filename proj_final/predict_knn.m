function predicted_label = predict_knn(X_train, Y_train, X_test)

k=20;

test_size=size(X_test,1);
train_size=size(X_train,1);

cos_sim = @(x,y) dot(x,y)./(norm(x,2)*norm(y,2));

op=zeros(test_size,1);
for i=1:test_size
    new_person = X_test(i,:);
    cos_sim_matrix=zeros(train_size,1);
    for j=1:train_size
        old_person=X_train(j,:);
        cos_sim_matrix(j)=cos_sim(new_person,old_person);
    end
    [sajal,sajal_idx] = sort(cos_sim_matrix,'descend');
    idx = sajal_idx(1:k,:); 
    gender = mean(Y_train(idx,:));
    if gender > 0.5
        gender=1;
    else
        gender=0;
    end
    op(i)=gender;
end

predicted_label=op;