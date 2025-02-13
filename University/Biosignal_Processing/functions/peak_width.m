function [t_i_0_idx,t_i_1_idx] = peak_width(yvals,peak_idx,peak_val,hight)
%peak_width calculates the two x indices where a peak has a certain y-value
%determined by "hight" at full width

%INPUTS:
%yvals          array of y values
%peak_idx       idex at with the peak of the array is
%peak_val       value of the peak
%hight          parameter to dtermine the width of the peak at a certain
%               hight (peak_lvl). From 0 to 1

peak_lvl = peak_val*hight;
t_i_0_idx = find(yvals(1:peak_idx) <= peak_lvl, 1, 'last'); %ending at the peak's index search for every idx where yvals is smaller or same as halfpeak, then select last one
t_i_1_idx = peak_idx + find(yvals(peak_idx:end) <= peak_lvl, 1, 'first'); %starting at the peak's index search for every idx where yvals is smaller or same as halfpeak, then select first one
t_i_1_idx = t_i_1_idx - 1; %remove one exessive index

end

