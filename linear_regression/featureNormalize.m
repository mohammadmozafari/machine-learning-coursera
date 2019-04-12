function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
[m, n] = size(X);
mu = zeros(1, n);
sigma = zeros(1, n);

for i = 1:n
    avg = 0;
    for j = 1:m
        avg = avg + X(j, i);
    end
    avg = avg / m;
    mu(i) = avg;
    sigma(i) = std(X_norm(:, i));
    X_norm(:, i) = X_norm(:, i) - avg;
    X_norm(:, i) = X_norm(:, i) / sigma(i);
end
