for i = 1:k
    X(i) = string(i);
end
bag_tr = bagOfWords(X, data_train(:, 1:500));
bag_tr = full(tfidf(bag_tr));
bag_te = bagOfWords(X, data_test(:, 1:500));
bag_te = full(tfidf(bag_te));
%%
data_train(:, 1:500) = bag_tr;
data_test(:, 1:500) = bag_te;