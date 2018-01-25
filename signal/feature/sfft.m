% This function produce short-time Fourier transform on the input signal
% according to the settings 
% Author: Xiao Xiong
% Created: 2006
% Last modified: 24 Jun, 20015


function fft_x = sfft(x,frame_size,frame_shift,Fs,FFT_length,window_type,do_DC_remove, useGPU)

x(:);   % convert x to a column vector
frame_overlap = frame_size - frame_shift;

if exist('do_DC_remove')==0 || length(do_DC_remove)==0
    do_DC_remove = 1;
end
if exist('useGPU')==0 || length(useGPU)==0
    useGPU = 0;
end

% produce the hamming windowm
if exist('window_type')==0 || length(window_type)==0
    window = my_hamming(frame_size);
else
    switch window_type
        case 'hamming'
            window = my_hamming(frame_size);
        case 'hanning'
            window = hanning(frame_size);
        case 'rectangular'
            window = ones(frame_size,1);
    end
end

if do_DC_remove
    % DC offset removing
    x = DC_remove(x,0.999);
    % pre-emphasis, boost the high frequency spectrum
    
    % Method 2: Direct operation. Fast
    y = x(2:end) - 0.97*x(1:end-1);
    x = [x(1); y];

%     % Method 1: Call filter function. Too slow
%     A    = [1 -0.97];
%     x = filter(A,1,x);
end

% number of blocks
N_block = floor((length(x)-frame_size)/frame_shift)+1;

% % Method 1: intuitive but slow
% fft_x = zeros(FFT_length,N_block);
% for i = 1:N_block
%     % step 1. framing
%     start = (i-1)*frame_shift+1;
%     last = min(length(x),(i-1)*frame_shift+frame_size);
%     x_fr = x(start:last);
%     % step 2. windowing
%     x_fr = x_fr(:).*window;
%     fft_x(:,i) = fft(x_fr,FFT_length);
% end

% Method 2: user buffer function and then call fft just once
needed_size = (N_block-1)*frame_shift + frame_size;
overlap = frame_size - frame_shift;

if license('test', 'Signal_Toolbox')
    x_store = buffer(x(overlap+1:needed_size),frame_size,overlap);
else
    x_store = enframe(x(1:needed_size), frame_size, frame_shift)';
end


x_store(:,1) = x(1:frame_size);
x_store(:,2) = x(frame_shift+1:frame_shift+frame_size);

if useGPU
    x_store = gpuArray(x_store);
end
x_store = bsxfun(@times, x_store, window);

fft_x = fft(x_store,FFT_length);

% Recover wav
if 0
    img = sqrt(-1);
    mag_x = abs(fft_x(1:size(fft_x,1)/2+1,:))';
    phase_x = angle(fft_x(1:size(fft_x,1)/2+1,:))';
    x_recon = mag_x .* ( cos(phase_x) + img*sin(phase_x));
    x_recon = conj(x_recon);
    x_recon(:,FFT_length/2+2:FFT_length) = conj(x_recon(:,FFT_length/2:-1:2));
    
    xi = ifft(x_recon');
    xi2 = ifft(fft_x);
    
    wav = my_ola(real(xi), frame_size, frame_size-frame_shift);
    
end
