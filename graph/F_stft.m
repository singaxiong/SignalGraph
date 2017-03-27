function [fft_x, maskFFT] = F_stft(input_layer, curr_layer)
input = input_layer.a;
[nCh,T,N] = size(input);

fft_len = curr_layer.fft_len;
frame_len = curr_layer.frame_len;
frame_shift = curr_layer.frame_shift;

win_type = ReturnFieldWithDefaultValue(curr_layer, 'win_type', 'hamming');
removeDC = ReturnFieldWithDefaultValue(curr_layer, 'removeDC', 1);
doDithering = ReturnFieldWithDefaultValue(curr_layer, 'doDithering', 1);
precision = class(gather(input(1)));
useGPU = IsInGPU(input);

if N==1
    fft_x = sfft_multi(input',frame_len,frame_shift,fft_len, win_type,removeDC, IsInGPU(input), doDithering);
    fft_x = fft_x(1:fft_len/2+1,:,:);
    [d1,d2,d3] = size(fft_x);
    fft_x = reshape(fft_x, d1*d2, d3);
    maskFFT = [];
else
    [mask, variableLength] = CheckTrajectoryLength(input);
    input2 = PadShortTrajectory(input, mask, 0);
    input2 = reshape(permute(input2, [2 1 3]), T, nCh*N);
    
    fft_x = sfft_multi(input2,frame_len,frame_shift,fft_len, win_type,removeDC, IsInGPU(input), doDithering);
    fft_x = fft_x(1:fft_len/2+1,:,:);
    fft_x = permute(fft_x, [1 3 2]);
    [nBin,nFr,d3] = size(fft_x);
    fft_x = reshape(fft_x, nBin, nFr, nCh, N);
    fft_x = permute(fft_x, [1 3 2 4]);
    fft_x = reshape(fft_x, nBin*nCh, nFr, N);
    
    % now build a mask for spectrogram
    nSampleChannel = sum(mask==0);
    nSampleChannel = gather(nSampleChannel);
    if useGPU
        maskFFT = gpuArray.zeros(nFr, N, precision);
    else
        maskFFT = zeros(nFr, N, precision);
    end
    for i=1:N
        nFrChannel = enframe_decide_frame_number(nSampleChannel(i), frame_len, frame_shift, 0);
        maskFFT(nFrChannel+1:end,i) = 1;
    end
end
end
