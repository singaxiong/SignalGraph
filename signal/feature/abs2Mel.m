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

function [X_mel] = abs2Mel(X_abs, Fs, N_mel)

if nargin < 2
    Fs = 8000;
end
if nargin < 3
    N_mel = 23;             % for AURORA2 database, 23 mel filterbank are used
end

[N_vec,n] = size(X_abs);
FFT_length = 2*n;

% generate the mel window
mel_win = mel_window_FE(N_mel, FFT_length/2, Fs);
% X_mel = zeros(N_vec,N_mel);
% for i=1:N_vec
%     X_mel(i,:) = X_abs(i,:)*mel_win;
% end
% faster implementation
X_mel = X_abs*mel_win;