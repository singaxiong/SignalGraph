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
        if isfield(reader, 'multiArrayFiles') && reader.multiArrayFiles
            for j=1:length(files{i})
                [tmp_wav fs] = audioread(files{i}{j});
                if j==1
                    wav = zeros(length(tmp_wav),length(files{i}));
                end
                wav(:,j) = tmp_wav;
            end
        else
            if isfield(reader, 'fs')
                [wav, fs] = Reader_waveform(files(1), reader.fs);
            else
                [wav, fs] = Reader_waveform(files(1));
            end
            wav = wav{1}';
        end
        if fs>8000
            FFT_length = 512;
        else
            FFT_length = 256;
        end
        if isfield(reader, 'useChannel')
            wav = wav(:, reader.useChannel);
        end
        nCh = size(wav,2);
        tmp = sfft_multi(wav,fs*0.025,fs*0.01,FFT_length, [], 0, 0);
        % [~,tmp] = wav2abs_multi(wav, fs, [], [], [], useGPU);
        nFFT = size(tmp,1);
        tmp = tmp(1:nFFT/2+1,:,:);
        if strcmpi(precision, 'single')
            tmp = single(tmp);
        end
        tmp_feat = reshape(tmp, nCh*(nFFT/2+1), size(tmp,3));
    else
        if isfield(reader, 'fs')
            [wav, fs] = Reader_waveform(files(i), reader.fs);
        else
            [wav, fs] = Reader_waveform(files(i));
        end
        [~,tmp] = wav2abs(wav{1}', fs, [], [], [], useGPU);
        nFFT = size(tmp,1);
        tmp_feat = tmp(1:nFFT/2+1,:);
    end
    if strcmpi(precision, 'single')
        feat{i} = single(gather(tmp_feat));
    else
        feat{i} = gather(tmp_feat);
    end
end
end