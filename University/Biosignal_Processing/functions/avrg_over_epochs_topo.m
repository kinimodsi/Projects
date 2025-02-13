function [average_potential_of_locations,time_diff] = avrg_over_epochs_topo(epoched_data,timepoint)
%calculate the average potential of each channel at one time point across epochs

    [time_diff, idx_time] = min(abs(epoched_data.times - timepoint)); % check for time in dataset nearest to the specified timepoint
    epoched_data_timepoint = epoched_data.data(:, idx_time, :); %extract that timepoint from data
    average_potential_of_locations = mean(epoched_data_timepoint,3); %take mean across all epochs for each channel for the timepont
end

