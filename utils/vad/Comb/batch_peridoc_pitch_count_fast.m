function [valFeature] = batch_peridoc_pitch_count_fast(testWave, Fs, numFrame, nSamplePerFrame, nSampleForward)
x_start   = 1;
x_end     = nSamplePerFrame;
act_frame_x = zeros(nSamplePerFrame*2,numFrame);
for j=1:numFrame
    act_frame_x(:,j) =  [testWave(x_start:x_end); zeros(x_end-x_start+1,1)];
    x_start = x_start + nSampleForward;
    x_end   = x_end   + nSampleForward;
end

[frSize, nFr] = size(act_frame_x);
binWidth = Fs/frSize;
hammWinFrame  = hamming(frSize);
HighPassFreqUpper = 800;
numHighPassFreqUpperBin = ceil(HighPassFreqUpper/binWidth);
mapX=[1:numHighPassFreqUpperBin];
mapX = 1/numHighPassFreqUpperBin.*mapX;
mapX = mapX';

frame_x = zeros(nSamplePerFrame*2,numFrame);
for i=1:nFr    
    frame_x(:,i) = act_frame_x(:,i)/norm(act_frame_x(:,i));
% so that signal level is NOT at play here!!!
end

frame_x_hamm  = bsxfun(@times, frame_x, hammWinFrame);
abs_fft_x = abs(fft(frame_x_hamm));
abs_fft_x(1:numHighPassFreqUpperBin,:) = bsxfun(@times, abs_fft_x(1:numHighPassFreqUpperBin,:), mapX);
abs_fft_x = bsxfun(@times, abs_fft_x, 1./max(abs_fft_x(1:2*HighPassFreqUpper/binWidth,:)));

% The above operation with mapX reduces low frequency power, basically
% frequency filtering!!!

% extract noise level from 2K-3K freq range
abs_fft_x = abs_fft_x.^1.2;

L = floor(frSize/2);
binWidth = (Fs/(L*2));
PitchRangeLower = 80;
PitchRangeUpper = 250;

PitchIdxRange = ceil(PitchRangeUpper - PitchRangeLower)/binWidth;
val_histPeak = [];
val_histTrough = [];

SumPeakVal = [];
SumTroughVal = [];
beamWidth =2;

for (i=1:PitchIdxRange)
    pitchVal  = floor( ((PitchRangeLower/binWidth) +i-1)*binWidth);
    
    % number of harmonics, skipping the first harmonics bcos very noisy!!!
    for (j=2:7)
        idx          = floor((j*pitchVal)/binWidth);
        idx_trough   = floor(idx + (pitchVal/(2*binWidth)));
        
        Pi  = sum(abs_fft_x(idx-beamWidth:idx+beamWidth,:));
        Ti =  sum(abs_fft_x(idx_trough-beamWidth:idx_trough+beamWidth,:));
        
        SumPeakVal(:,i,j-1)   =  Pi;
        SumTroughVal(:,i,j-1) = Ti;
    end
    
end
SP = sum(SumPeakVal,3);
ST = sum(SumTroughVal,3);
val_histTrough  = ST.^2;

for i=1:PitchIdxRange
    currSumPeakVal = squeeze(SumPeakVal(:,i,:))';
    currSumTroughVal = squeeze(SumTroughVal(:,i,:))';
    val_histPeak(:,i)   = SP(:,i).^2 - var(currSumPeakVal)' - var(currSumTroughVal)';
end

[V1 iV1] = max(val_histPeak');
for i=1:nFr
    V2(i) = abs(val_histTrough(i,iV1(i)));
end
valFeature      = V1./V2;
