% Form the mel triangle window when given the number of mel bins and the number of linear
% frequency bins

function [window] = mel_window( N_mel_bin, N_linear_freq_bin, linear_samp )
% N_linear_freq_bin is the number of frequency bins used by FFT, e.g. 256

FFT_length = N_linear_freq_bin*2;

% find the mel bin centers
[bin_centers] = mel_center_FE(linear_samp, N_mel_bin );

% round off the mel bin centers to the linear freq bins
rounded_bin_centers = round( bin_centers/linear_samp*2* (N_linear_freq_bin) );
ctr(1) = 0;
ctr(2:N_mel_bin+1) = rounded_bin_centers;
ctr(N_mel_bin+2) = N_linear_freq_bin-1;

window = zeros(N_linear_freq_bin, N_mel_bin+2);
% window(:,i) represents window for the k=i-1 winsow
% window(i,:) represents i-1 th item of a paticular window
% window for the k=0 
for i=1:ctr(2)-ctr(1)
    window(i,1) = 1- (i-1)/( ctr(2)-ctr(1) );
end

% window for k=N_mel_bin + 1 
for i=(ctr(N_mel_bin+1)+2) : ctr(N_mel_bin+2)+1
    window(i,N_mel_bin+2) = (i-1-ctr(N_mel_bin+1)) / (ctr(N_mel_bin+2)-ctr(N_mel_bin+1));
end

% window for k=1:N_mel_bin
for i=2:N_mel_bin+1
    for j=1+ctr(i-1):ctr(i)+1
        window(j,i) = (j-1-ctr(i-1)) / (ctr(i)-ctr(i-1));
    end
    for j=2+ctr(i):ctr(i+1)+1
        window(j,i) = 1-(j-1-ctr(i)) / (ctr(i+1)-ctr(i));
    end
end
