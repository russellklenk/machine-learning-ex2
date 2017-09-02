function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features
minv = 1/m;
lambda_over_2m  = lambda / (2*m);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
theta_sub = theta(2:end);
reg_term  =(lambda / (2*m)) * sum(theta_sub .^ 2);
h         = sigmoid(X * theta);
J         = (minv * (-y' * log(h) - (1-y)' * log(1 - h)));
J         = J + reg_term;
rt_g      =((lambda / m) * theta);
rt_g(1)   = 0;
grad      =(minv * X' * (h - y)) + rt_g;





% =============================================================

end
