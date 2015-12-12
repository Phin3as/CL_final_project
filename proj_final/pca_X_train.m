function [pca_X_train,pca_X_test] = do_pca(X_train, X_test,numpc)

X_full = [X_train;X_test];
train_size=size(X_train,1);

[ev pv eigenvalues] = pca(X_full);

ev_size=numel(eigenvalues);

if (numpc==0)
    numpc=1000;
end

pca_X_train = pv(1:train_size,1:numpc);
pca_X_test = pv(train_size+1:end,1:numpc);