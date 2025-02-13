function [epoched_data_channel_epoavrg, time_vector] = avrg_over_epochs_time(epoched_data, channels)
    %takes the cross epoch average over time and takes root-mean square across the channels

    % returns data and time vector
    
    % get channel indicies
    idx_channels = find(ismember({epoched_data.chanlocs.labels}, channels));
    
    % extract data for the specified channels
    epoched_data_channels = epoched_data.data(idx_channels, :, :);

    epoched_data_channel_mean = rms(epoched_data_channels, 1); % Average over channels
    epoched_data_channel_epoavrg = squeeze(mean(epoched_data_channel_mean, 3)); % Average over epochs
   
    % Extract time vector
    time_vector = epoched_data.times; 
end