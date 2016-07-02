function fft_x = F_stft(input, curr_layer)
fft_len = curr_layer.fft_len;
frame_len = curr_layer.frame_len;
frame_shift = curr_layer.frame_shift;

win_type = ReturnFieldWithDefaultValue(curr_layer, 'win_type', 'hamming');
removeDC = ReturnFieldWithDefaultValue(curr_layer, 'removeDC', 1);

fft_x = sfft_multi(input,frame_len,frame_shift,fft_len, win_type,removeDC, IsInGPU(input));
fft_x = fft_x(1:fft_len/2+1,:,:);
[d1,d2,d3] = size(fft_x);
fft_x = reshape(fft_x, d1*d2, d3);

end
