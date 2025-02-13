%****************************************************************************************************
%
%   Shrinkage LDA: TEST
%
%   Author: Stefan Ehrlich
%   Last revised: 12.12.2014
%
%   Input:
%   - model based on trainShrinkLDA
%   - featv
%   - labels [-1, 1]
%   Output:
%   - model
%
%****************************************************************************************************

function [y] = predictShrinkLDA(model,featv)

    y = double(sign(model.w*featv'+model.b)); % labels [-1 1]
    
    for i = 1:length(y)
        if y(i) == -1
            y(i) = model.labelscodes(1);
        elseif y(i) == +1
            y(i) = model.labelscodes(2);
        end
    end

end
