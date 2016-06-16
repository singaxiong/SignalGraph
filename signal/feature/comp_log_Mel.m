% comp_log_Mel receives a time domain speech signal and produces its
% log Mel filterbank coefficients. It is the Matlab counterpart of the
% feature extraction program provided with AURORA2 database. 
% Author: Xiao Xiong
% Created: 18 Jul, 2005
% Last Modified: 28 Jul, 2005
% Inputs: 
%       x   1-D time domain signal
% Outputs: 
%       hist_log_mel    the log Mel filterbank coefficients
%       hist_mel        the Mel filterbank coefficients
%       hist_abs        the spectral magnitude
%       logE            the log energy item needed in computing MFCC

function [hist_log_mel, hist_mel, hist_abs, hist_fft,hist_x, logE] = comp_log_Mel(x);

x(:);   % convert x to a column vector

FFT_length = 256;
logE_floor = -50;           % minimum value for log energy
Fs = 8000;              % for AURORA2 database, the sampling rate is 8k
N_mel = 23;             % for AURORA2 database, 23 mel filterbank are used

frame_size = Fs * 0.025;    % 25ms frame
frame_shift = Fs * 0.01;    % 10ms shift
frame_overlap = frame_size - frame_shift;

x_old = x;
% DC offset removing
x = DC_remove(x,0.999);
% pre-emphasis, boost the high frequency spectrum
A    = [1 -0.97];
x = filter(A,1,x);    % y(n) = x(n) - 0.97*x(n-1)
% number of blocks
N_block = floor((length(x)-frame_size)/frame_shift)+1;
% generate the mel window
mel_win = mel_window_FE(N_mel, FFT_length/2, Fs);

% produce the hamming windowm
window = hamming(frame_size);

    hist_x = zeros(frame_size,N_block);
    hist_fft = zeros(FFT_length,N_block);
    hist_abs = zeros(FFT_length/2+1,N_block);
    hist_mel = zeros(N_mel,N_block);
    hist_log_mel = zeros(N_mel,N_block);
    
for i = 1:N_block
%     if mod(i,100) == 0, disp(i); end;
    % step 1. framing
    start = (i-1)*frame_shift+1;
    last = min(length(x),(i-1)*frame_shift+frame_size);
    x_fr = x(start:last);
    if i<N_block, hist_x(:,i) = x_fr; end
    % step 2. calculate the log energy
    logE(i) = log(max(logE_floor, sum(x_old(start:last).^2)));
    % step3. windowing
    x_fr = x_fr(:).*window;
    % step 4. zero padding, if the number of elements in x_fr is less than
    % the length of FFT, append zeros to its end
    x_fr = [x_fr' zeros(1,FFT_length-length(x_fr))];
    % step 5, calculate the Fourier transform using Fast Fourier Transform
    X = fft(x_fr(:),FFT_length);
    % step 6. extract the magnitude of the spectral coefficients
    X_abs = abs(X(2:FFT_length/2+1)); 
    % step 7. mel-scale window wraping
    X_mel = X_abs'*mel_win;
    % step 8. take the natural logarithm
    X_log_mel = log(X_mel);
    
    
    hist_fft(:,i) = X;
    hist_abs(:,i) = abs(X(1:FFT_length/2+1));
    hist_mel(:,i) = X_mel;
    hist_log_mel(:,i) = X_log_mel;
end
%figure;
%imageFE(hist_log_mel');
a=1;