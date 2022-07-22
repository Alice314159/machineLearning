function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%hx = X * theta;
%J = sum((hx - y).^2)/2/m +lambda*sum(theta.^2)/2/m

%grad = sum(X'*(hx-y))/m + theta*lambda/m
%grad(1) = sum(X'*(hx-y))/m 


h=X*theta;
theta(1,1)=0;
%线性回归代价函数
J=(sum(power((h-y),2))+lambda*sum(power(theta,2)))/(2*m);

%梯度下降
grad=((h-y)'*X).*(1/m)+(theta').*(lambda/m);





% =========================================================================

grad = grad(:);

end
