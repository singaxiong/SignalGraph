function layer = InitMelLayer(layer, para)

Mel_idx = ReturnLayerIdxByName(layer, 'Mel');
MelWindow = mel_window_FE(para.topology.nFbank, para.topology.fft_len/2, para.topology.fs)';
MelWindow(:,end+1) = 0;
layer{Mel_idx}.W = MelWindow;
layer{Mel_idx}.b = zeros(para.topology.nFbank,1);

end
