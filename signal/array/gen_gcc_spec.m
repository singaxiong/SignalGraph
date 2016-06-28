% This function generate spectrogram and GCC for microphone array input
function [gcc, magnitude, phase, gcc_interp] = gen_gcc_spec(wav, para)
fs = para.fs;

% compute the spectrogram
magnitude = {}; phase = {};
if para.genSpec
    specWinSize = para.specWinSize;
    specShift = para.specShift;
    for j=1:size(wav,1)
        [~, tmp] = wav2abs(wav(j,:)', fs, specShift, specWinSize);
        magnitude{j} = abs(tmp(1:size(tmp,1)/2+1,:))';
        phase{j} = angle(tmp(1:size(tmp,1)/2+1,:))';
    end
end

% compute the spatial correlation
gcc = []; gcc_interp = [];
if para.genGCC
    gccWinSize = para.gccWinSize;
    gccOverlap = para.gccOverlap;
    [correlationVectors]=getCorrelationVector_fast2(wav, gccWinSize*fs, gccOverlap/gccWinSize);
    
    if isfield(para, 'gcc_bin_range') ==0  % note that we only need a small part of the GCC related to microphone array topology and sampling rate
        para.gcc_bin_range = 10;
    end
    gcc = correlationVectors(gccWinSize*fs/2-para.gcc_bin_range:gccWinSize*fs/2+para.gcc_bin_range,:,:);
    if size(wav,1)>2
        gcc = permute(gcc, [1,3,2]);
        [d1,d2,d3] = size(gcc);
        gcc = reshape(gcc, d1*d2, d3);
    end
    gcc = gcc';

    if para.genSpec     % align GCC to spec
        if size(gcc,1)==0
            gcc_interp = [];
            fprintf('error: no gcc is extracted\n');
        elseif size(gcc,1)==1
            gcc_interp = repmat(gcc,size(magnitude{1},1),1);
        else
            fr_ctr_gcc = [1:size(gcc,1)]*(gccWinSize-gccOverlap);
            fr_ctr_spec = [1:size(magnitude{1},1)]*specShift;
            gcc_interp = interp1(fr_ctr_gcc, gcc, fr_ctr_spec);
        end
        
        % find the first non NaN value of gcc
        ISNAN = isnan(gcc_interp(:,1));        
        for j=1:length(ISNAN)
            if ISNAN(j)==0; break; end
        end
        gcc_interp(1:j-1,:) = repmat(gcc_interp(j,:), j-1,1);
        
        % find the last non NaN value of gcc
        ISNAN = isnan(gcc_interp(:,1));
        for j=length(ISNAN):-1:1
            if ISNAN(j)==0; break; end
        end
        gcc_interp(j+1:end,:) = repmat(gcc_interp(j,:), length(ISNAN)-j,1);
    end
end

end
