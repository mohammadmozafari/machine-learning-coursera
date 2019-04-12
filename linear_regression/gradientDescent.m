function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
t = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    temp = zeros(t, 1);
    for j = 1:t
        der = 0;
        for i = 1:m
            der = der + ((theta' * X(i, :)') - y(i)) * X(i, j);
        end
        der = der / m;
        temp(j) = theta(j) - alpha * der;
    end
    for j = 1:t
        theta(j) = temp(j);
    end
    J_history(iter) = computeCost(X, y, theta);
end 

end
