function averaged_channels = channel_average(EEG_data, channels)
%calculates the root-mean-square value of multiple channels 

    %get indices of the specified channels
    idx_channels = find(ismember({EEG_data.chanlocs.labels}, channels));
    
    % Initialize averaged_channels
    num_epochs = size(EEG_data.data, 3);
    num_samples = size(EEG_data.data, 2);
    averaged_channels = zeros(num_epochs, num_samples);
    
    % Calculate the average across specified channels for each epoch
    for i = 1:num_epochs
        epoch_data = EEG_data.data(idx_channels, :, i);
        averaged_channels(i, :) =  rms(epoch_data, 1);
    end
end