function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
H = sigmoid(X * theta);
yT = y';
theta(1) = 0;
 
J = (-1/m) * (yT * log(H) + (1 - y)' * log(1 - H));
J = J + (lambda / (2 * m)) * (theta' * theta);

grad = (1 / m) * (X' * (H - y));
grad = grad + (lambda / m) * theta;

end
