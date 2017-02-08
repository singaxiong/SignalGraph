function layer = genNetworkFbankExtraction(para)
nFreqBin = para.fft_len/2+1;

layer{1}.name = 'Input';
layer{end}.inputIdx = 1;
layer{end}.dim = [1 1];

layer{end+1}.name = 'stft';
layer{end}.prev = -length(layer)+1;
layer{end}.fft_len = para.fft_len;
layer{end}.frame_len = para.frame_len;
layer{end}.frame_shift = para.frame_shift;
layer{end}.removeDC = para.removeDC;
layer{end}.win_type = para.win_type;
layer{end}.dim = [(para.fft_len/2+1) layer{length(layer)+layer{end}.prev}.dim(1)];
layer{end}.skipBP = 1;  % skip backpropagation

layer{end+1}.name = 'Affine';       % scaling the Fourier transform
layer{end}.prev = -1;
layer{end}.W = [];
layer{end}.b = [];
layer{end}.dim = [1 1] * layer{length(layer)+layer{end}.prev}.dim(1);
layer{end}.update = 0;
layer{end}.skipBP = 1;  % skip backpropagation

% get the log power spectrum and perform CMN
layer{end+1}.name = 'Power';
layer{end}.prev = -1;
layer{end}.dim = [1 1] * layer{end-1}.dim(1);
layer{end}.skipBP = 1;

layer{end+1}.name = 'Mel';
layer{end}.prev = -1;
layer{end}.W = [];
layer{end}.b = [];
layer{end}.dim = [para.nFbank nFreqBin];
layer{end}.update = 0;

layer{end+1}.name = 'Log';
layer{end}.const = 1e-2;
layer{end}.prev = -1;
layer{end}.dim = [1 1]*layer{end-1}.dim(1);
layer{end}.skipBP = 1;

layer{end+1}.name = 'CMN';
layer{end}.prev = -1;
layer{end}.dim = [1 1]*layer{end-1}.dim(1);
layer{end}.skipBP = 1;

layer = FinishLayer(layer);

end
