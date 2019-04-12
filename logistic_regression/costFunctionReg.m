function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
H = sigmoid(X * theta);
yT = y';

J = (-1/m) * (yT * log(H) + (1 - y)' * log(1 - H));
J = J + ((lambda / (2 * m)) * (theta' * theta - theta(1)^2));

grad = zeros(length(theta), 1);
grad(1) = (1/m) * X(:, 1)' * (H - y);
grad(2:end) = (1/m) * (X(:, 2:end)' * (H - y)) + (lambda/m)*(theta(2:end));

end
