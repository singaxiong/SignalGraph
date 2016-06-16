% Compute the log energy of the speech signal from the time domain samples
% Note that: to compute LogE, there is no need to do DC_offset removing and
% Preemphasis. However, my experiments shows that DC offset removing and
% preemphasis in LogE computation improves the recognition accuracy quite a
% lot
function logE = comp_logE(x);

x(:);   % convert x to a column vector

logE_floor = -50;           % minimum value for log energy
Fs = 8000;              % for AURORA2 database, the sampling rate is 8k

frame_size = Fs * 0.025;    % 25ms frame
frame_shift = Fs * 0.01;    % 10ms shift
frame_overlap = frame_size - frame_shift;

% number of blocks
N_block = floor((length(x)-frame_size)/frame_shift)+1;
    
for i = 1:N_block
    % step 1. framing
    start = (i-1)*frame_shift+1;
    last = min(length(x),start+frame_size-1);
    x_fr = x(start:last);
    % step 2. calculate the log energy
    logE(i) = log(max(logE_floor, sum(x_fr.^2)));
end