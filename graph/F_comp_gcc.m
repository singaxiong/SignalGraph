function gcc = F_comp_gcc(input, curr_layer)
frame_len = curr_layer.frame_len;
frame_shift = curr_layer.frame_shift;

[nSample, nCh] = size(input);

gcc_dim = curr_layer.dim(1) / (nCh*(nCh-1)/2);

[gcc]=getCorrelationVector_fast2(input', frame_len, 1 - frame_shift/frame_len, IsInGPU(input));

gcc_bin_range = (gcc_dim-1)/2;

gcc = gcc(frame_len/2-gcc_bin_range:frame_len/2+gcc_bin_range,:,:);

if nCh>2
    gcc = permute(gcc, [1,3,2]);
    [d1,d2,d3] = size(gcc);
    gcc = reshape(gcc, d1*d2, d3);
end

end
