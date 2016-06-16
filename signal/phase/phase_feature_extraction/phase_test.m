function [GDD_phase, MGD_phase] = phase_test(x, Fs)
frame_size = 0.025*Fs;
frame_shift = 0.01*Fs;
FFT_length = 512;
do_DC_remove = 1;

% complex spectrogram extraction
x(:);   % convert x to a column vector
frame_overlap = frame_size - frame_shift;

window = hamming(frame_size);

if do_DC_remove
    % DC offset removing
    x = DC_remove(x,0.999);
    % pre-emphasis, boost the high frequency spectrum
    
    % Method 2: Direct operation. Fast
    y = x(2:end) - 0.97*x(1:end-1);
    x = [x(1); y];
end

% number of blocks
N_block = floor((length(x)-frame_size)/frame_shift)+1;

% Method 2: user buffer function and then call fft just once
needed_size = (N_block-1)*frame_shift + frame_size;
overlap = frame_size - frame_shift;
x_store = buffer(x(overlap+1:needed_size),frame_size,overlap);
x_store(:,1) = x(1:frame_size);
x_store(:,2) = x(frame_shift+1:frame_shift+frame_size);
x_store = x_store.*repmat(window,1,N_block);
fft_x = fft(x_store,FFT_length);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% delay spectrogram extraction
delay_vector = [1:frame_size]';
delay_matrix = repmat(delay_vector, 1, N_block);

delay_x = x_store .* delay_matrix;
delay_fft_x = fft(delay_x,FFT_length);

fft_x = fft_x(1:FFT_length/2+1,:)';
phase_clean = angle(fft_x');
delay_fft_x = delay_fft_x(1:FFT_length/2+1,:)';

GDD_phase = group_delay_feature(sp_complex); % deviation along frequency aixs
MGD_phase = modified_group_delay_raw(sp_complex, sp_delay);

