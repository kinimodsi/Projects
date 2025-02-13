%****************************************************************************************************
%
%   Shrinkage LDA: TRAIN 
%
%   Author: Stefan Ehrlich
%   Last revised: 12.12.2014
%
%   Input:
%   - featv
%   - labels [-1, 1]
%   - lambda (if lambda = NaN, analytic shrinkage will be performed)
%   Output:
%   - model
%
%****************************************************************************************************

function model = trainShrinkLDA(featv,labels,lambda)

    model.labelscodes = unique(labels);

    x1 = featv(labels==model.labelscodes(1),:);
    x2 = featv(labels==model.labelscodes(2),:);

    mu1 = mean(x1,1);
    mu2 = mean(x2,1);

    % shrinkage
    
    if isequal(lambda, 'analytic') % perform analytic shrinkage
        [model.lambda1, cov1] = lambda_estimate(x1);
        [model.lambda2, cov2] = lambda_estimate(x2);

    elseif isequal(lambda, 0)
        cov1 = (1-lambda)*cov(x1);
        cov2 = (1-lambda)*cov(x2);
    else
        cov1 = (1-lambda)*cov(x1)+lambda*eye(size(x1,2));
        cov2 = (1-lambda)*cov(x2)+lambda*eye(size(x2,2));
    end

    model.w = (mu2-mu1)/(cov1+cov2);
    model.b = -model.w*(mu1+mu2)'/2;

end