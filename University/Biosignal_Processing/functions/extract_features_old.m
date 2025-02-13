function [feature_matrix,label_vector] = extract_features_old(epoched_data, label, timepoints)
% function to extract the feature matrix and label vector of epoched data.
% Features consist of the amplitude of each eeg channel at different
% timepoints specified in "timepoints"

%INPUT:
% epoched_data;     epoched eeg data already event specific
% label;            label of the epoched data. When no labels present, input 0 
% % timepoints;       timepoints at which the amplitude is used as a feature


%OUTPUT:
% feature_matrix;   matrix of dimensions: (channels*timepoints) x events
% label_vector;     vector of (the same) label for each feature


idx_times = zeros(1, length(timepoints)); % initiate index vector

for i = 1:length(timepoints)
        [~, idx_times(i)] = min(abs(epoched_data.times - timepoints(i))); %get time indices for each time point
end

%initiate feature matrix:
num_channels = length(epoched_data.chanlocs); % extract number of channels
feature_vector_lenght= num_channels*length(idx_times);
event_count=length(epoched_data.data(1,1,:));
feature_matrix= zeros(event_count,feature_vector_lenght);


for l = 1:num_channels  %create feature matrix frim channel and time indicies
    idx_channels = find(strcmp({epoched_data.chanlocs(l).labels}, {epoched_data.chanlocs.labels}));
    for i = 1:length(idx_times)
        for event = 1:event_count
        column_index = (l - 1) * length(idx_times) + i;
        feature_matrix(event,column_index) = epoched_data.data(idx_channels, idx_times(i),event);
        end
    end
end

%return label vector with label from input or NaN for label=0
if label==0
    label_vector=NaN(event_count, 1);
else    
    label_vector=ones(event_count, 1)*label;
end

