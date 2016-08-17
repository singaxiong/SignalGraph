% reader struct
%    array: specify whether input is array or single channel
%    multiArrayFiles: if input is array, specify whether each channel is
%    one file. 
%    useChannel: if input is array, specify which channel(s) to use.
%    if not defined, use all channels. 
%
function feat = Reader_wavfile2spectrum(files, reader, useGPU, precision)
if nargin<4
    precision = 'single';
end
for i=1:length(files)
    if reader.array
        [wav, fs] = Reader_waveform(files(i), reader);
		if strcmpi(precision, 'single')
			wav = single(wav{1});
		else
			wav = double(wav{1});
		end

        reader = SetDefaultValue(reader, 'frame_len', fs*0.025);
        reader = SetDefaultValue(reader, 'frame_shift', fs*0.01);
        reader = SetDefaultValue(reader, 'window_type', 'hamming');
        reader = SetDefaultValue(reader, 'removeDC', 0);
        reader = SetDefaultValue(reader, 'useGPU', 0);
        
        FFT_length = 2^nextpow2(reader.frame_len);
        nCh = size(wav,2);
        tmp = sfft_multi(wav, reader.frame_len, reader.frame_shift, FFT_length, reader.window_type, reader.removeDC, reader.useGPU);
        % take the first half of the Fourier coefficients
        tmp = tmp(1:FFT_length/2+1,:,:);
        % reshape multi-channel Fourier coefficients into a vector
        tmp_feat = reshape(tmp, nCh*(FFT_length/2+1), size(tmp,3));
    end
    if strcmpi(precision, 'single')
        feat{i} = single(gather(tmp_feat));
    else
        feat{i} = double(gather(tmp_feat));
    end
end
end