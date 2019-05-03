function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
H = X * theta;

% computes the linear regression regularized cost function and gradients
theta(1) = 0;
J = (H - y)' * (H - y) / (2 * m);
J = J + lambda * (theta' * theta) / (2 * m);

grad = ((X' * (H - y)) + lambda * theta) / m;

end
