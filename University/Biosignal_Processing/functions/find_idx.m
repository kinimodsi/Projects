function [index] = find_idx(list, value)
    % calculate the differences between each element and the desired value
    differences = abs(list - value);
    %find index of the minimum difference
    [~, index] = min(differences);
end