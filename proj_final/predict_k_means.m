function predicted_label = predict_k_means(X_train, Y_train, X_test)

k=20;

cos_sim = @(x,y) dot(x,y)./(norm(x,2)*norm(y,2));

cluster_idx = kmeans(X_train,k);
clusters=zeros(k,size(X_train,2));
labels=zeros(k,1);

for i=1:k
    cluster_data = (cluster_idx==i);
    actual_cluster_data = X_train(cluster_data,:);
    clusters(i,:) = mean(actual_cluster_data(:,:));
    labels(i,:) = mean(Y_train(cluster_data,:));
end

%clusters is the centroid of each cluster
%K means classisification is done at this point

for i=1:size(X_test,1)
    new_person = X_test(i,:);
    cos_sim_matrix=zeros(numel(labels),1);
    for j=1:k
        old_cluster = clusters(j,:);
        cos_sim_matrix(j)=cos_sim(old_cluster,new_person);
    end
    [sajal,sajal_idx] = sort(cos_sim_matrix,'descend');
    idx = sajal_idx(1:1,:); %num of closest clusters to pick
    gender = mean(labels(idx,:));
    if gender > 0.5
        gender=1;
    else
        gender=0;
    end
    op(i)=gender;
end

predicted_label=op';