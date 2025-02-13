function [feature_matrix, label_vector] = extract_features(epoched_data, label, peak_locs, peak_locs_fft, Fs, peak_locs_feat_spacial)
% function to extract the feature matrix and label vector of epoched data.
% Features consist of the average amplitude of each eeg channel at
% certain ranges around the peaks and specific frequency features

% INPUT:
% epoched_data;             epoched eeg data already event specific
% label;                    label of the epoched data. When no labels present, input 0 

% peak_locs;                contains information about the time indices of the peeks width 
%                  

% peaks_fft;                contains information about the peaks in the frequency domain.
%                           (frequency_idx_1, frequency_1;
%                           frequency_idx_2, frequency_2)

% Fs;                       sampling frequency for fft

%peak_locs_feat_spacial;    contains information about the peaks of the spatial features


% OUTPUT:
% feature_matrix;   matrix of dimensions: (events x [channels*(peaks + fft_peaks)+spatial features])
% label_vector;     vector of labels

num_peaks = size(peak_locs, 1); % get the number of peaks
num_peaks_fft = size(peak_locs_fft, 1); % get the number of fft_peaks
num_peaks_spacial = size(peak_locs_feat_spacial,1) ;% get the number of spatial peaks
event_count = size(epoched_data.data, 3); % get number of events
num_channels = length(epoched_data.chanlocs); % extract number of channels

%initial feature matrix
feature_vector_length = num_channels * (num_peaks + num_peaks_fft) + num_peaks_spacial; 
feature_matrix = zeros(event_count, feature_vector_length);

for l = 1:num_channels  %create temporal features from channel and time indices
    for i = 1:num_peaks
        for event = 1:event_count
            column_index = (l - 1) * (num_peaks + num_peaks_fft) + i;
            average_amplitude = mean(epoched_data.data(l, peak_locs(i, 1):peak_locs(i, 2), event), 'all');
            feature_matrix(event, column_index) = average_amplitude;
        end
    end
end

% Add frequency domain features
for l = 1:num_channels
    for j = 1:num_peaks_fft
        for event = 1:event_count
            column_index = (l - 1) * (num_peaks + num_peaks_fft) + num_peaks + j;
            epoched_data_fft=fft_ss(epoched_data.data(l,:,event),Fs);
            epoched_data_fft = mov_avrg_custom(epoched_data_fft,3,2);
            average_frequency_amplitude = mean(epoched_data_fft(peak_locs_fft(j, 1):peak_locs_fft(j, 2)), 'all');
            feature_matrix(event, column_index) = average_frequency_amplitude;
        end
    end
end
% indicies for spacial feature:
idx_channel_Cz = find(strcmp({epoched_data.chanlocs.labels}, "Cz"));
idx_channel_O2 = find(strcmp({epoched_data.chanlocs.labels}, "O2"));
idx_channel_F3 = find(strcmp({epoched_data.chanlocs.labels}, "F3"));
idx_channel_Fp2 = find(strcmp({epoched_data.chanlocs.labels}, "Fp2"));
idx_channel_P4 = find(strcmp({epoched_data.chanlocs.labels}, "P4"));

%central-edge, F3-O2 and Fp2-P4 features:
channels_c=["Cz", "CP1", "CP2", "FC1", "FC2"];
channels_e=["O1", "O2", "P8"];
for event = 1:event_count
            %calculate mag. diffrences for each event
            central = channel_average(epoched_data, channels_c);
            edge = channel_average(epoched_data, channels_e);

            %calculate single channel amplitudes:
            %Cz=epoched_data.data(idx_channel_Cz, :, event);
            O2=epoched_data.data(idx_channel_O2, :, event);
            F3=epoched_data.data(idx_channel_F3, :, event);
            Fp2=epoched_data.data(idx_channel_Fp2, :, event);
            P4=epoched_data.data(idx_channel_P4, :, event);


            %diff_mag_Cz_O2=Cz - O2;
            diff_mag_c_e = central(event, :) - edge(event, :); % use averaged channels for one feature
            diff_mag_F3_O2= O2 - F3;   
            diff_mag_Fp2_P4 = Fp2 - P4;
            %write features into feature matrix:
            column_index = num_channels * (num_peaks + num_peaks_fft) + 1;
            average_amplitude_c_e = mean(diff_mag_c_e(peak_locs_feat_spacial(1,1):peak_locs_feat_spacial(1,2)), 'all');
            feature_matrix(event, column_index) = average_amplitude_c_e;

            column_index2 = num_channels * (num_peaks + num_peaks_fft) + 2;
            average_amplitude_F3_O2 = mean(diff_mag_F3_O2(peak_locs_feat_spacial(2,1):peak_locs_feat_spacial(2,2)), 'all');
            feature_matrix(event, column_index2) = average_amplitude_F3_O2;

            column_index3 = num_channels * (num_peaks + num_peaks_fft) + 3;
            average_amplitude_Fp2_P4 = mean(diff_mag_Fp2_P4(peak_locs_feat_spacial(3,1):peak_locs_feat_spacial(3,2)), 'all');
            feature_matrix(event, column_index3) = average_amplitude_Fp2_P4;

end

%return label vector with label from input or NaN for label=0
if label == 0
    label_vector = NaN(event_count, 1);
else    
    label_vector = ones(event_count, 1) * label;
end
end