function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
H = [ones(m, 1), X];
H = sigmoid(H * Theta1');
H = [ones(m, 1), H];
H = sigmoid(H * Theta2');
I = [1:size(H, 1)]';
% H is a m * 10 matrix

% Computing the regularized cost function
J = 0;
newH = H(sub2ind(size(H), I, y));
J = J + sum(log(newH));
logCom = log(1 - H);
logCom(sub2ind(size(H), I, y)) = 0;
J = J + sum(sum(logCom));
J = (-1 / m) * J;

sum1 = sum(sum(Theta1(:, 2:end) .^ 2));
sum2 = sum(sum(Theta2(:, 2:end) .^ 2));
RegTerm = (lambda / (2 * m)) * (sum1 + sum2);
J = J + RegTerm;
% J is computed by now

y_vector = zeros(num_labels, 1);
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for t = 1:m

    % Perform forward propagation
    a_1 = X(t, :)';
    a_1 = [1; a_1];
    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);

    % Perform back prop to compute delta_3 and delta_2
    y_vector(y(t)) = 1;
    delta_3 = a_3 - y_vector;
    y_vector(y(t)) = 0;
    delta_2 = ((Theta2)' * delta_3) .* sigmoidGradient([1; z_2]);
    delta_2 = delta_2(2:end);

    % Accumulate Delta
    Delta_1 = Delta_1 + delta_2 * a_1';
    Delta_2 = Delta_2 + delta_3 * a_2';

end

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
Theta1_grad = (Delta_1 + lambda * Theta1) / m;
Theta2_grad = (Delta_2 + lambda * Theta2) / m;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
