function [x_butt,b,a] = butter_custom(x_vals,order,f_cutoff,f_sample,filter_function,filter_kind)
%custom butter filter where you can specify order, kind and wheather you do
%only forward or bidirectional filtering.
    arguments
        x_vals (1,:) {mustBeNumeric, mustBeFinite};
        order   double 
        f_cutoff    double 
        f_sample    double 
        filter_function (1,:) char {mustBeMember(filter_function,{'filter','filtfilt'})}
        filter_kind (1,:) char {mustBeMember(filter_kind,{'low','high','bandpass'})}
    end
    [b,a] = butter(order,f_cutoff/(f_sample/2),filter_kind); % use butter filter function to get transfer function coeffitients
    if filter_function == "filtfilt"
        x_butt = filtfilt(b,a,x_vals);%filtfilt to filter forth and back -> no shifting of data, but not suitable for realtime data
    elseif filter_function == "filt"
        x_butt = filter(b,a,x_vals); %filter to filter only forth for real time and computational expensive data but is delayed
    else
        error("No such filter function. Try filt or filfilt for filter_function")
    end
    %fvtool(b,a);
end