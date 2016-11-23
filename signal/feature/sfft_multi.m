% This function produce short-time Fourier transform on the input signal
% according to the settings 
% Author: Xiao Xiong
% Created: 2006
% Last modified: 24 Jun, 2006


function fft_x = sfft_multi(x,frame_size,frame_shift,FFT_length,window_type,do_DC_remove, useGPU, doDithering)

% x should be a TxN matrix, where T is the number of samples, and N is the
% number of channels. 

if exist('do_DC_remove')==0 || length(do_DC_remove)==0
    do_DC_remove = 1;
end
if exist('useGPU')==0 || length(useGPU)==0
    useGPU = 0;
end

if exist('doDithering')==0 || length(doDithering)==0
    x = x + randn(size(x))/2^32;
end

% produce the hamming windowm
if exist('window_type')==0 || length(window_type)==0
    window = my_hamming(frame_size);
else
    switch window_type
        case 'hamming'
            window = hamming(frame_size);
        case 'hanning'
            window = hanning(frame_size);
        case 'rectangular'
            window = ones(frame_size,1);
    end
end

if do_DC_remove
    % DC offset removing
    x = DC_remove(gather(x),0.999);
    % pre-emphasis, boost the high frequency spectrum

    if 0
        % Method 2: Direct operation. In multichannel case, method 2 is
        % slower
        y = x(2:end,:) - 0.97*x(1:end-1,:);
        x = [x(1,:); y];
    else
        % Method 1: Call filter function. Too slow
        A    = [1 -0.97];
        x = filter(A,1,gather(x));
    end
end

if useGPU && ~IsInGPU(x)
    x = gpuArray(x);
end
x_store = my_enframe(x, frame_size, frame_shift);
x_store = bsxfun(@times, x_store, window);

fft_x = fft(x_store,FFT_length);

