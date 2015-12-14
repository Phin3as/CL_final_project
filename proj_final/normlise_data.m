n=size(X_train,1)

for i=1:n
    x_row=X_train(i,:);
    den = norm(x_row);
    X_train(i,:) = x_row./den;
end

n=size(X_test,1)

for i=1:n
    x_row=X_test(i,:);
    den = norm(x_row);
    X_test(i,:) = x_row./den;
end