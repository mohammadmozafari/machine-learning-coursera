function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(m, 1), X];
p = X * Theta1';
p = sigmoid(p);
p = [ones(m, 1), p];
p = p * Theta2';
p = sigmoid(p);
[~, p] = max(p, [], 2);

end
