clear all; close all; clc;
%% import data and functions 
Fs=256; %sampling frequency after down_sampling
addpath("functions"); %add folder with functions to path
calibration = pop_loadset('filename', 'calibration.set', 'filepath', 'Data'); %load data
%priont channel locations:
%fprintf('channels_locs (exampls): %s\n', calibration.chanlocs.labels);
%% Epoching

%epoch data based on tamewindow defined by epoch_start and epoch_end
epoch_start = -0.5;
epoch_end = 0.65;
EEG_epo_nonError = pop_epoch(calibration,{"S  4"},[epoch_start, epoch_end]); % 
EEG_epo_Error = pop_epoch(calibration,{"S  5"},[epoch_start, epoch_end]); %  

%% filtering and baseline

%for idx_channel=1:27 
    %EEG_epo_nonError.data(idx_channel, :) = butter_custom(EEG_epo_nonError.data(idx_channel, :),2,1.5,Fs,"filtfilt","high");
    %EEG_epo_Error.data(idx_channel, :) = butter_custom(EEG_epo_Error.data(idx_channel, :),2,1.5,Fs,"filtfilt","high");
    %EEG_epo_nonError.data(idx_channel, :) = mov_avrg_custom(EEG_epo_nonError.data(idx_channel, :),9,8);
    %EEG_epo_Error.data(idx_channel, :) = mov_avrg_custom(EEG_epo_Error.data(idx_channel, :),9,8);
%end

%remove single trial baseline:
EEG_epo_nonError = pop_rmbase(EEG_epo_nonError, [epoch_start*1000, -250],[]);  %remove the single-trial ERP baseline
EEG_epo_Error = pop_rmbase(EEG_epo_Error, [epoch_start*1000, -250],[]);


%% plot potential average across epochs (ERP) to compare error and nonError (avergaed over central channels):
%define central channels:
channels_central=["Cz", "CP1", "CP2", "FC1", "FC2"];

[EEG_epo_nonError_central_epo_avrg, t] = avrg_over_epochs_time_mean(EEG_epo_nonError,channels_central);
EEG_epo_Error_central_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_Error,channels_central);

%% plot every epoch of channel central channels to get overview

figure; % initiate plot
for i=1:size(EEG_epo_Error.data, 3) %loop through error data and plot 
    averaged_channels = channel_average(EEG_epo_Error,channels_central);
    plot(t,averaged_channels(i,:),'Color',[1, 0, 0, 0.2],LineWidth=0.005);
    alpha(.5);
    hold on;
end
%plot and assign name to last line for legend:
last_error_line = plot(t,averaged_channels(i,:),'Color',[1, 0, 0, 0.2],LineWidth=0.005); 


for i=1:size(EEG_epo_nonError.data, 3) %loop through non-error data and plot 
    averaged_channels = channel_average(EEG_epo_nonError,channels_central);
    plot(t,averaged_channels(i,:),'Color',[0, 0, 1, 0.2],LineWidth=0.005);
    alpha(.5);
    hold on;
end
%plot and assign name to last line for legend:
last_nonError_line = plot(t,averaged_channels(i,:),'Color',[0, 0, 1, 0.2],LineWidth=0.005);

hold off;
legend([last_error_line,last_nonError_line],"Error","Non Error")
title("All Epochs of central channels (Cz, CP1, CP2, FC1, FC2) for overview")

%% compute and plot fft (all and average) of epochs channel Cz
%define parameters for fft:
idx_channel_Cz = find(strcmp({EEG_epo_nonError.chanlocs.labels}, "Cz")); %getting index from channel CZ
N = length(EEG_epo_Error.data(idx_channel_Cz, :, 1));

%define arrays to store fft values in to calculate the average later
fft_average_Error = zeros(1, N/2 +1);
fft_average_nonError = zeros(1, N/2 +1);

%start plot:
figure;
for i = 1:size(EEG_epo_Error.data, 3) %get data, calculate fft (error)
    epoched_data_channel = EEG_epo_Error.data(idx_channel_Cz, :, i);
    [fft_epoched_data, freqs]= fft_ss(epoched_data_channel,Fs);

    %smoothen data via moving average filter:
    fft_epoched_data = mov_avrg_custom(fft_epoched_data,3,2); 

    %plot:
    plot(freqs, fft_epoched_data,'Color',[1, 0, 0, 0.2],"LineWidth", 0.005);
    hold on;

    fft_average_Error = fft_average_Error + fft_epoched_data; %add magnitude to average array
end

for i = 1:size(EEG_epo_Error.data, 3) %get data, calculate fft (non-error)
    epoched_data_channel = EEG_epo_nonError.data(idx_channel_Cz, :, i);
    [fft_epoched_data, freqs]= fft_ss(epoched_data_channel,Fs);

    %smoothen data via moving average filter
    fft_epoched_data = mov_avrg_custom(fft_epoched_data,3,2); 
    
    %plot:
    plot(freqs, fft_epoched_data,'Color',[0, 0, 1, 0.2],"LineWidth", 0.005);
    hold on;
    
    fft_average_nonError = fft_average_nonError + fft_epoched_data; %add magnitude to average array
end
xlim([0 40]); %define x range for plot

% Calculate the average FFT magnitudes for error and non-error
fft_average_Error = fft_average_Error / size(EEG_epo_Error.data, 3);
fft_average_nonError = fft_average_nonError / size(EEG_epo_nonError.data, 3);

%calculate the fifference between error and non-error average
difference_magnitude_freq = fft_average_Error-fft_average_nonError;

%% select peaks in frequency domain:

% find peaks and sort via "findpeaks" and "sort" functions:
[peaks_fft, peak_locs_fft] = findpeaks(difference_magnitude_freq, 'MinPeakProminence', 0.02, 'WidthReference', 'halfheight'); %only select peaks in higher freqs
[sorted_peaks_fft, sorted_idx_fft] = sort(peaks_fft,"descend");

% get indices of interesting peaks:
f_1_idx = peak_locs_fft(sorted_idx_fft(1)); 
f_1=freqs(f_1_idx);
range_fft=1; % define little range around the peak to average across
f_1_0=f_1_idx-range_fft;
f_1_1=f_1_idx+range_fft;
peak_locs_f_feat=[f_1_0, f_1_1]; %only use highest peak since it looks like its the only peak containing nice information

% Plot the average FFT magnitudes and the difference between error and non_error
hold on;
fft_average_Error_line = plot(freqs, fft_average_Error, 'red', 'LineWidth', 1);
fft_average_nonError_line = plot(freqs, fft_average_nonError, 'blue', 'LineWidth', 1);
difference_magnitude_freq_line = plot(freqs, difference_magnitude_freq, 'black', 'LineWidth', 1);

%mark frequency peaks in plot:
plot([f_1,f_1],[0,difference_magnitude_freq(f_1_idx)],'black--',linewidth=1);

% plot peak_freqs_width:
plot([freqs(f_1_0),freqs(f_1_1)],[1,1],'black',linewidth=1);

%define plot:
xlabel('Frequency (Hz)');
xlim([0 40]);
ylabel('Magnitude');
title('(Average) FFT Magnitudes for Error and non-Error across Epochs');
legend([fft_average_Error_line,fft_average_nonError_line,difference_magnitude_freq_line], "Error FFT average", "Non-Error FFT average","Difference magnitude Error/NonError");
grid on;
hold off;

%% select peaks in time domain:
%calculate difference between error/nonError:
difference_magnitude=abs(EEG_epo_Error_central_epo_avrg-EEG_epo_nonError_central_epo_avrg);

%define windows to search in:
window_start1 = find_idx(t, 150);
window_end1 = find_idx(t, 355);
window_start2 = find_idx(t, 370);
window_end2 = find_idx(t, 620);

%get peak information
[peaks1, peak_locs1] = findpeaks(difference_magnitude(window_start1:window_end1), 'MinPeakProminence', 0.01, 'WidthReference', 'halfheight');
peak_locs1=peak_locs1+window_start1-1;
[peaks2, peak_locs2] = findpeaks(difference_magnitude(window_start2:window_end2), 'MinPeakProminence', 0.01, 'WidthReference', 'halfheight');
peak_locs2=peak_locs2+window_start2-1;


% sorting peaks
[sorted_peaks1, sorted_idx1] = sort(peaks1,"descend");
[sorted_peaks2, sorted_idx2] = sort(peaks2,"descend");

% Select idx's of highest peaks
%window1:
t_1_idx = peak_locs1(sorted_idx1(1));
t_2_idx = peak_locs1(sorted_idx1(2));
%window2:
t_3_idx = peak_locs2(sorted_idx2(1));
t_4_idx = peak_locs2(sorted_idx2(2));
t_5_idx = peak_locs2(sorted_idx2(3));

%get values for width of the peaks:
%window1:
hight1=0.4; %0.4 works well
hight2=0.5;
[t_1_0_idx,t_1_1_idx]=peak_width(difference_magnitude,t_1_idx,sorted_peaks1(1),hight1);
[t_2_0_idx,t_2_1_idx]=peak_width(difference_magnitude,t_2_idx,sorted_peaks1(2),hight2);

%window2:
hight3=0.8; %0.8 works well
hight4=0.7;
hight5=0.6;
[t_3_0_idx,t_3_1_idx]=peak_width(difference_magnitude,t_3_idx,sorted_peaks2(1),hight3);
[t_4_0_idx,t_4_1_idx]=peak_width(difference_magnitude,t_4_idx,sorted_peaks2(2),hight4);
[t_5_0_idx,t_5_1_idx]=peak_width(difference_magnitude,t_5_idx,sorted_peaks2(3),hight5);


%get time values from indices for plotting: (there might be a more elegant way)
t_1=t(t_1_idx);
t_2=t(t_2_idx);
t_3=t(t_3_idx);
t_4=t(t_4_idx);
t_5=t(t_5_idx);
t_1_0 = t(t_1_0_idx);
t_1_1 = t(t_1_1_idx);
t_2_0 = t(t_2_0_idx);
t_2_1 = t(t_2_1_idx);
t_3_0 = t(t_3_0_idx);
t_3_1 = t(t_3_1_idx);
t_4_0 = t(t_4_0_idx);
t_4_1 = t(t_4_1_idx);
t_5_0 = t(t_5_0_idx);
t_5_1 = t(t_5_1_idx);

%store peak ranges in array for feature extraction:
peak_locs_feat=[t_1_0_idx, t_1_1_idx;
                t_2_0_idx, t_2_1_idx;
                t_3_0_idx, t_3_1_idx;
                t_4_0_idx, t_4_1_idx;
                t_5_0_idx, t_5_1_idx]; 

%% plot averaged ERPs (error, nonError and mag. difference) over time:

figure;
plot(t,EEG_epo_nonError_central_epo_avrg,LineWidth=1);
hold on;
plot(t,EEG_epo_Error_central_epo_avrg,LineWidth=1);
hold on;
plot(t,difference_magnitude,LineWidth=2,Color="black");
hold on;

%mark peaks:
plot([t_1,t_1],[0,difference_magnitude(t_1_idx)],'black--',linewidth=1);
plot([t_2,t_2],[0,difference_magnitude(t_2_idx)],'black--',linewidth=1);
plot([t_3,t_3],[0,difference_magnitude(t_3_idx)],'black--',linewidth=1);
plot([t_4,t_4],[0,difference_magnitude(t_4_idx)],'black--',linewidth=1);
plot([t_5,t_5],[0,difference_magnitude(t_5_idx)],'black--',linewidth=1);

%mark selection area
area([t(window_start1),t(window_end1)],[10,10],"FaceColor","black",'FaceAlpha', 0.1);
area([t(window_start2),t(window_end2)],[10,10],"FaceColor","black",'FaceAlpha', 0.1);
text(-500,8,{"gray areas mark regions","for peaks to search in"},'FontSize',10)

% plot peak widths:
plot([t_1_0,t_1_1],[sorted_peaks1(1)*hight1,sorted_peaks1(1)*hight1],'black',linewidth=1);
plot([t_2_0,t_2_1],[sorted_peaks1(2)*hight2,sorted_peaks1(2)*hight2],'black',linewidth=1);
plot([t_3_0,t_3_1],[sorted_peaks2(1)*hight3,sorted_peaks2(1)*hight3],'black',linewidth=1);
plot([t_4_0,t_4_1],[sorted_peaks2(2)*hight4,sorted_peaks2(2)*hight4],'black',linewidth=1);
plot([t_5_0,t_5_1],[sorted_peaks2(3)*hight5,sorted_peaks2(3)*hight5],'black',linewidth=1);

% define plot:
grid("on")
xlabel('time locked to key press');
ylabel('${\mu}V$','interpreter','latex');
title('Averaged ERP time course of central channels');
legend("nonError","Error","Difference","Peaks");

%% calculate the average potential of each channel at one time point across epochs
[t_sorted] = sort([t_1,t_2]);  %sort peaks in time for the case the first peak is larger than second

%calculate for error / nonError and p3 / n2 peaks:
EEG_epo_nonError_avrg_topo_n2 = avrg_over_epochs_topo(EEG_epo_nonError,t_sorted(1)); 
EEG_epo_nonError_avrg_topo_p3 = avrg_over_epochs_topo(EEG_epo_nonError,t_sorted(2));
EEG_epo_Error_avrg_topo_n2 = avrg_over_epochs_topo(EEG_epo_Error,t_sorted(1));
EEG_epo_Error_avrg_topo_p3 = avrg_over_epochs_topo(EEG_epo_Error,t_sorted(2));

%display topographic maps:
figure;
subplot(2,2,1);
topoplot(EEG_epo_nonError_avrg_topo_n2,calibration.chanlocs,'maplimits',[-4 4]);
title('noError N2')

subplot(2,2,2);
topoplot(EEG_epo_Error_avrg_topo_n2,calibration.chanlocs,'maplimits',[-4 4]);
title('Error N2')

subplot(2,2,3);
topoplot(EEG_epo_nonError_avrg_topo_p3,calibration.chanlocs,'maplimits',[-4 4]);
title('noError P3')

subplot(2,2,4);
topoplot(EEG_epo_Error_avrg_topo_p3,calibration.chanlocs,'maplimits',[-4 4]);
title('Error P3')
%% visualize potential spacial features
% calculate the average ERP of diffrent channels across epochs
[EEG_epo_nonError_Cz_epo_avrg, t] = avrg_over_epochs_time_mean(EEG_epo_nonError,"Cz");
EEG_epo_Error_Cz_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_Error,"Cz");
EEG_epo_nonError_O2_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_nonError,"O2");
EEG_epo_Error_O2_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_Error,"O2");
EEG_epo_nonError_F3_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_nonError,"F3");
EEG_epo_Error_F3_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_Error,"F3");
EEG_epo_nonError_Fp2_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_nonError,"Fp2");
EEG_epo_Error_Fp2_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_Error,"Fp2");
EEG_epo_nonError_P4_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_nonError,"P4");
EEG_epo_Error_P4_epo_avrg = avrg_over_epochs_time_mean(EEG_epo_Error,"P4");

%store central and lower right edge channels in arrays
channels_c=["Cz", "CP1", "CP2", "FC1", "FC2"]; %central channels
channels_e=["O1", "O2", "P8"]; %channels lower edge

% calculate spatial average over central/edge channels (error/nonError):
central_E = avrg_over_epochs_time(EEG_epo_Error,channels_c); 
edge_E= avrg_over_epochs_time(EEG_epo_Error,channels_e); 
central_nE = avrg_over_epochs_time(EEG_epo_nonError,channels_c);
edge_nE= avrg_over_epochs_time(EEG_epo_nonError,channels_e);

% calculate diffrence central/edge
diff_central_edge_Error = central_E-edge_E;
diff_central_edge_nonError = central_nE-edge_nE;

% calculate diffrence error/nonError:
diff_diff_central_edge=abs(diff_central_edge_nonError-diff_central_edge_Error);

% calculate difference between certain channels (error/nonError):
%Cz-O2
diff_Cz_O2_nonError=EEG_epo_nonError_Cz_epo_avrg-EEG_epo_nonError_O2_epo_avrg;
diff_Cz_O2_error=EEG_epo_Error_Cz_epo_avrg-EEG_epo_Error_O2_epo_avrg;

%F3-O2
diff_F3_O2_nonError=EEG_epo_nonError_F3_epo_avrg-EEG_epo_nonError_O2_epo_avrg;
diff_F3_O2_error=EEG_epo_Error_F3_epo_avrg-EEG_epo_Error_O2_epo_avrg;

%Fp2-P4
diff_Fp2_P4_nonError=EEG_epo_nonError_Fp2_epo_avrg-EEG_epo_nonError_P4_epo_avrg;
diff_Fp2_P4_error=EEG_epo_Error_Fp2_epo_avrg-EEG_epo_Error_P4_epo_avrg;

% calculate diffrence error/nonError between certain channels:
diff_diff_Cz_O2=abs(diff_Cz_O2_nonError-diff_Cz_O2_error);
diff_diff_F3_O2=abs(diff_F3_O2_nonError-diff_F3_O2_error);
diff_diff_Fp2_P4=abs(diff_Fp2_P4_nonError-diff_Fp2_P4_error);

%plot channel differences and central/edge difference as across epoch average over time
figure;
hold on;
plot(t,diff_diff_Cz_O2,LineWidth=1,Color="red");
plot(t,diff_diff_F3_O2,LineWidth=1,Color="green");
plot(t,diff_diff_Fp2_P4,LineWidth=1, Color="blue");
plot(t,diff_diff_central_edge,LineWidth=1,Color="black");
grid on;
xlabel('time locked to key press');
ylabel('${\mu}V$','interpreter','latex');
title('averaged ERP time course differences between channels and error/nonError');

%% select peaks in spatial features:
%get peak information:
[peaks_Cz_O2, peak_locs_Cz_O2] = findpeaks(diff_diff_Cz_O2, 'MinPeakProminence', 1, 'WidthReference', 'halfheight');
[peaks_F3_O2, peak_locs_F3_O2] = findpeaks(diff_diff_F3_O2, 'MinPeakProminence', 1, 'WidthReference', 'halfheight');
[peaks_Fp2_P4, peak_locs_Fp2_P4] = findpeaks(diff_diff_Fp2_P4, 'MinPeakProminence', 0.5, 'WidthReference', 'halfheight');

% sorting peaks
[sorted_peaks_Cz_O2, sorted_idx_Cz_O2] = sort(peaks_Cz_O2,"descend");
[sorted_peaks_F3_O2, sorted_idx_F3_O2] = sort(peaks_F3_O2,"descend");
[sorted_peaks_Fp2_P4, sorted_idx_Fp2_P4] = sort(peaks_Fp2_P4,"descend");

% Select idx's of highest peaks
t_Cz_O2_idx = peak_locs_Cz_O2(sorted_idx_Cz_O2(1));
t_F3_O2_idx = peak_locs_F3_O2(sorted_idx_F3_O2(1));
t_Fp2_P4_idx = peak_locs_Fp2_P4(sorted_idx_Fp2_P4(1));

%get values for width of the peaks:
hight_Cz_O2=0.3; 
hight_F3_O2=0.5; 
hight_Fp2_P4=0.5;  

%save peak information for feature extraction:
[t_Cz_O2_0_idx,t_Cz_O2_1_idx]=peak_width(diff_diff_Cz_O2,t_Cz_O2_idx,sorted_peaks_Cz_O2(1),hight_Cz_O2);
[t_F3_O2_0_idx,t_F3_O2_1_idx]=peak_width(diff_diff_F3_O2,t_F3_O2_idx,sorted_peaks_F3_O2(1),hight_F3_O2);
[t_Fp2_P4_0_idx,t_Fp2_P4_1_idx]=peak_width(diff_diff_Fp2_P4,t_Fp2_P4_idx,sorted_peaks_Fp2_P4(1),hight_Fp2_P4);
t_Cz_O2=t(t_Cz_O2_idx);
t_F3_O2=t(t_F3_O2_idx);
t_Fp2_P4=t(t_Fp2_P4_idx);
t_Cz_O2_0 = t(t_Cz_O2_0_idx);
t_Cz_O2_1 = t(t_Cz_O2_1_idx);
t_F3_O2_0 = t(t_F3_O2_0_idx);
t_F3_O2_1 = t(t_F3_O2_1_idx);
t_Fp2_P4_0 = t(t_Fp2_P4_0_idx);
t_Fp2_P4_1 = t(t_Fp2_P4_1_idx);

%store peak range indices for feature extraction:
peak_locs_feat_spacial=[t_Cz_O2_0_idx, t_Cz_O2_1_idx;
                t_F3_O2_0_idx, t_F3_O2_1_idx;
                t_Fp2_P4_0_idx, t_Fp2_P4_1_idx;
                ];

% disp(peak_locs_feat_spacial);

%plot peaks:
plot([t_Cz_O2,t_Cz_O2],[0,diff_diff_Cz_O2(t_Cz_O2_idx)],'red--',linewidth=1);
plot([t_F3_O2,t_F3_O2],[0,diff_diff_F3_O2(t_F3_O2_idx)],'green--',linewidth=1);
plot([t_Fp2_P4,t_Fp2_P4],[0,diff_diff_Fp2_P4(t_Fp2_P4_idx)],'blue--',linewidth=1);
plot([t_Cz_O2_0,t_Cz_O2_1],[sorted_peaks_Cz_O2(1)*hight_Cz_O2,sorted_peaks_Cz_O2(1)*hight_Cz_O2],'red--',linewidth=1);
plot([t_F3_O2_0,t_F3_O2_1],[sorted_peaks_F3_O2(1)*hight_F3_O2,sorted_peaks_F3_O2(1)*hight_F3_O2],'green--',linewidth=1);
plot([t_Fp2_P4_0,t_Fp2_P4_1],[sorted_peaks_Fp2_P4(1)*hight_Fp2_P4,sorted_peaks_Fp2_P4(1)*hight_Fp2_P4],'blue--',linewidth=1);
legend('Differnce between channels and labels Cz-O2","Difference F3-O2","Difference Fp2-P4","Difference central-edge');
hold off;
disp('plotting done')

%% Feature extraction:

% extract the temporal, spectral and spatial features and label vector.
% More information in function describtion of "extract_features"
[features_nE,labels_nE] = extract_features(EEG_epo_nonError,-1,peak_locs_feat, peak_locs_f_feat, Fs, peak_locs_feat_spacial);
[features_E,labels_E] = extract_features(EEG_epo_Error,1,peak_locs_feat,peak_locs_f_feat, Fs, peak_locs_feat_spacial);

% put together feature matrices and label vectors vertically
feature_matrix=[features_nE; features_E]; 
label_vector=[labels_nE; labels_E];
label_vector_boost=[labels_nE; labels_E]; %copy lable vector for boosting methods

%normalize feature matrix:
feature_matrix = zscore(feature_matrix);

disp(['size featurematrix: ', num2str(size(feature_matrix))]);
disp(['size labelvecor: ', num2str(size(label_vector))]);

%rank features according to fisher criterium
[d,rank]=fisherrank(feature_matrix,label_vector);
disp(['best feature strenght: ', num2str(d(rank(1)))])
%% Feature evaluation

% show fisher scores of the features
figure;
bar(d(rank));  
xlabel('Feature Index (Ranked)');
ylabel('Fisher Score');
title('Fisher Scores of Features');

%select top two features
top_two_features_indices = rank(1:2);
top_two_features = feature_matrix(:, top_two_features_indices);
% get indicices for each class
class_1_indices = label_vector == -1;
class_2_indices = label_vector == 1;

% scatter plot of best two feature to visualize class separability
figure;
scatter(top_two_features(class_1_indices, 1), top_two_features(class_1_indices, 2), 'b', 'filled');  
hold on;
scatter(top_two_features(class_2_indices, 1), top_two_features(class_2_indices, 2), 'r', 'filled'); 
xlabel('1. best feature');
ylabel('2. best feature');
legend('Class -1', 'Class 1');
hold off;


%% Feature selection 
%only use features with fisher score above threshhold:
threshold = 0.16;  %0.1 - 0.2 works good 0.5 already gives accuracy of 0.892 

% Select features with Fisher score above the threshold
best_features_indices = rank(d(rank) > threshold);
feature_matrix = feature_matrix(:, best_features_indices);
disp(['Selected amount of features: ', num2str(size(feature_matrix, 2))]);

%% CV and evaluation
% Model parameters and CV setup
kernel_func = "linear"; %define kernel function
label_vector(label_vector == -1) = 0; % Convert -1 to 0 since classperf expects non-negative numbers

%values for cross-validation:
start_c=0.05;
end_c=30;
steps=100;
C_vec = logspace(log10(start_c),log10(end_c),steps); % values for regularization parameter C
num_folds = 10; %amount of cv folds (10% of data is hidden for performance evaluation)

% Initialize variables to store results across multiple CV iterations
number_CVs = 5;
best_C_values = zeros(1, number_CVs);
best_f1_scores = zeros(1, number_CVs);
accuracy = zeros(1, number_CVs);

disp(['Number of CV iterations: ', num2str(number_CVs)]);
disp(['Number of C values to check: ', num2str(steps)]);

%Initiate plot:
figure;
for cv_iteration = 1:number_CVs %loop though cv iterations
    performance = zeros(length(C_vec), 5); %initiate matrix to store evaluation metrics

    % Stratified cross-validation indices
    cv = cvpartition(label_vector, 'Kfold', num_folds, 'Stratify', true);

    for i = 1:length(C_vec)  % loop though different hyperparameters "C" for CV
        C = C_vec(i);

        %initialize performance meassuring object
        cp = classperf(label_vector);

        % Perform cross-validation
        for fold = 1:num_folds 
            % Get indices for training and testing data for the current fold
            trainIndices = training(cv, fold);
            testIndices = test(cv, fold);

            %train model on training set
            svm_model = fitcsvm(feature_matrix(trainIndices, :), label_vector(trainIndices), 'Standardize', true, 'KernelFunction', kernel_func, "KernelScale", "auto", 'BoxConstraint', C);

            %predict labels for test set
            predictedLabels = predict(svm_model, feature_matrix(testIndices, :));

            % update class performance:
            classperf(cp, predictedLabels, testIndices);
        end
        %store performance values of the cv iteration in performance matrix
        performance(i, 1) = cp.ErrorRate;
        performance(i, 2) = cp.CorrectRate;
        performance(i, 3) = cp.PositivePredictiveValue; % Precision
        performance(i, 4) = cp.Sensitivity; % Recall
        performance(i, 5) = harmmean([cp.PositivePredictiveValue, cp.Sensitivity]); % F1-score
    end
    %display progress:
    disp(['Iteration ', num2str(cv_iteration), ' of ', num2str(number_CVs), ' done']);

    %retrieve best C, f1-score and accuracy for current iteration based on f1-score
    [~, best_C_index] = max(performance(:, 5));
    best_C_values(cv_iteration) = C_vec(best_C_index);
    best_f1_scores(cv_iteration) = performance(best_C_index, 5);
    accuracy(cv_iteration) = performance(best_C_index,2);
    
    %plot f1-score and mcr over C for current CV iteration:
    yyaxis left
    ylim([0,1]);
    semilogx(C_vec, performance(:,1),"blue-");%errorrate
    ylabel('MCR');
    hold on;
    yyaxis right
    semilogx(C_vec, performance(:,5),"red-"); %f_1 score
    ylabel('F1-score');
    xlabel('C');
end
%add axis limits
ylim([0,1]);
xlim([start_c,end_c]);

% Calculate average and standard deviation of best C and evaluation metrics
avg_best_C = mean(best_C_values);
std_best_C = std(best_C_values);
avg_best_f1 = mean(best_f1_scores);
std_best_f1 = std(best_f1_scores);
avg_accuracy = mean(accuracy);
std_accuracy = std(accuracy);
avg_mcr = mean(1-accuracy);
std_mcr = std(1-accuracy);

%plot average best C:
avg_best_C_line = plot([avg_best_C,avg_best_C],[0,avg_best_f1],"red--");

%finish plot:
legend(avg_best_C_line, 'C for average best F1-score','Location','southwest');
title('F1 Score and Missclassificationrate over C values for different CV iterations');

% Display results
disp(['Average best F1-Score: ', num2str(avg_best_f1), ' +- ', num2str(std_best_f1)]);
disp(['Average best C: ', num2str(avg_best_C), ' +- ', num2str(std_best_C)]);
disp(['Average accuracy: ', num2str(avg_accuracy), ' +- ', num2str(std_accuracy)]);
disp(['Average Missclassificationrate: ', num2str(avg_mcr), ' +- ', num2str(std_mcr)]);


%% train model with best C and plot confusionmatrix
best_C=11.2828; % use C from values stated in the report

%train svm model:
svm_model = fitcsvm(feature_matrix, label_vector, 'Standardize', true, 'KernelFunction', kernel_func, "KernelScale", "auto", 'BoxConstraint', best_C);

%predict labels on train data:
predictions = predict(svm_model, feature_matrix);

%change labels back from 0 to -1:
predictions(predictions == 0) = -1;
label_vector(label_vector == 0) = -1;

%show confusion matrix
figure;
cm = confusionchart(label_vector,predictions,'Title','Confusion matrix train data');
% title("Confusion matrix train data");
MCR_train = sum(label_vector == predictions)/length(label_vector);
disp(['Accuracy on train data (not meaningful): ', num2str(MCR_train)]);

%% Testing on real data with the SVM model
%load recall data:
recall_data = pop_loadset('filename', 'recall.set', 'filepath', 'Data');

%preprocessing:
recall_data_epoched = pop_epoch(recall_data,{"S  6"},[epoch_start, epoch_end]);

%feature extraction and normalization:
features_rc = extract_features(recall_data_epoched,0,peak_locs_feat,peak_locs_f_feat, Fs, peak_locs_feat_spacial);
features_rc=zscore(features_rc);

% Select the same features as in training:
features_rc = features_rc(:, best_features_indices);
features_rc = zscore(features_rc);

%estimate labels using svm_model:
label_vector_rc = predict(svm_model, features_rc);

%change labels back from 0 to -1:
label_vector_rc(label_vector_rc == 0) = -1;

%rename label vector and save as .mat:
label_vector = label_vector_rc;
save("label_vector.mat", "label_vector");

%calculate Error / nonError to check plausibility of the result
detected_error_ratio=length(label_vector_rc(label_vector_rc == 1))/length(label_vector_rc);
disp(['Ratio Error / nonError: ', num2str(detected_error_ratio)]);

%% Test different boosting methods:
max_weak_classifiers = 30; % Maximum number of weak classifiers (doesnt improve after 50 max_classiferes)
models_names = {"GentleBoost", "AdaBoostM1"}; % compare multiple boosting methods
num_folds = 10; % Number of folds for cross-validation
max_depth = 5; % max tree depht
models = cell(length(models_names), 1); % Define cell array to store models

% Initialize array to store f1 scores:
f1_scores = zeros(length(models_names), max_weak_classifiers);

% Convert -1 to 0 since classperf expects non-negative numbers:
label_vector_boost(label_vector_boost == -1) = 0; 

disp('Trying boosting methods...')
% try different numbers of weak classifiers

for m = 1:length(models_names)
    disp("Testing " + models_names{m} + " with " + num2str(max_weak_classifiers) + " weak classifiers");
    for num_weak_classifiers = 1:max_weak_classifiers
        %initialize cv partitions
        cv = cvpartition(label_vector_boost, "KFold", num_folds, 'Stratify', true);
        % CV and F1 score calculation
        %initialize performance meassuring object
        cp = classperf(label_vector_boost);
        f1_scores_fold = zeros(num_folds, 1); %to store F1 scores for each fold
    
        % Perform cross-validation
        for fold = 1:num_folds
            train_indices = training(cv, fold);
            test_indices = test(cv, fold);
    
            % train model on training set with specified max depth
            t = templateTree('MaxNumSplits', max_depth);
            model = fitensemble(feature_matrix(train_indices, :), label_vector_boost(train_indices), models_names{m}, num_weak_classifiers, t);
            
            % predict on test set
            predictions = predict(model, feature_matrix(test_indices, :));
    
            % calculate F1 score
            cp = classperf(label_vector_boost(test_indices), predictions);
            f1_scores_fold(fold) = harmmean([cp.PositivePredictiveValue, cp.Sensitivity]);
        end
        models{m}=model;
        % Average F1 score across all folds
        avg_f1_score = mean(f1_scores_fold);
        f1_scores(m, num_weak_classifiers) = avg_f1_score;
    end
end


%plot F1 scores for each max depth value and model
figure;
hold on;
for m = 1:length(models_names)
    plot(1:max_weak_classifiers, f1_scores(m, :));
end
xlabel("Number of Weak Classifiers");
ylabel("F1 Score");
title("F1 Score over Number of Weak Classifiers");
legend("GentleBoost", "AdaBoost");
hold off;

%% Show confusion matrix of ensemble methods:
for m = 1:length(models_names)
    %predict labels with current model:
    predictions_e = predict(models{m}, feature_matrix);

    %change back 0 to -1 in label and prediction vector:
    predictions_e(predictions_e == 0) = -1;
    label_vector_boost(label_vector_boost == 0) = -1;
    
    %plot confusion matrix for each model:
    figure;
    confusionchart(label_vector_boost, predictions_e);
    title(["Confusion matrix train data of ", models_names{m}]);
end
%% Label estimation Ensemble:

% predict an recall dataset:
label_vector_rc_ensemble = predict(models{1}, features_rc);
label_vector_rc_ensemble(label_vector_rc_ensemble == 0) = -1;

%calculate Error / nonError to check plausibility of the result:
detected_error_ratio_ensemble = sum(label_vector_rc_ensemble == 1) / length(label_vector_rc_ensemble);
disp(['Ratio Error / nonError (gentle boost): ', num2str(detected_error_ratio_ensemble)]);