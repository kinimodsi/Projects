function [normalized_singlesided_fft2, freqs] = fft_ss(data,Fs)
% performs fft on data "data" with sampling frequency "Fs" 

% returns the normalized, single sided frequency spectrum
% "normalized_singlesided_fft2" and the frequency vector "freqs"

    N=length(data);       %signal_length  
    freqs=Fs*(0:(N/2))/N; %freq vector
    data_fft=fft(data,N); % does fft
    normalized_fft = abs(data_fft/N);         %normalize frequency components
    normalized_singlesided_fft = normalized_fft(1:N/2+1); %only consider first half, including DC and NQ freq
    normalized_singlesided_fft(2:end-1) = 2*normalized_singlesided_fft(2:end-1); %double magnitude to accound for redundant half
    normalized_singlesided_fft2=normalized_singlesided_fft; 
end

