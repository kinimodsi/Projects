function [x_ma] = mov_avrg_custom(x_vals,win_size,overlap)
%calculates the moving average of a signal "x_vals" via windowsize
%"win_size" and overlap "overlap"
    x_ma = zeros(1, length(x_vals)); % output array
    step_size = win_size - overlap; %calculate stepsize
    for i = 1:step_size:length(x_vals)-win_size+1 %perform moving average
        x_ma(i:i+win_size-1) = mean(x_vals(i:i+win_size-1));
    end
end
