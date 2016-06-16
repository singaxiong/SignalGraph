function [tauFull, ampFull, td, ld] = binaural_ft(micSig,fs,frameLength,overlap,micDist)
% PURPOSE : time delay and amplitude ratio feature for pairwise mic signals 
%
% INPUT   : micSig      , pairwise microphone signals, L*2.
%           fs          , sampling frequency.
%           frameLength , length of each speech frame, preferably set as 2^N.
%           overlap     , overlap of two consecutive frames.
%           micDist     , the distance of two microphones to compute the
%                         cut-off frequency.
% OUTPUT  : tauDiffFull , matrix of delay over different freqs.
%           ampDiffFull , matrix of amplitude ratio over different freqs.
%           td          , estimated time delay.
%           ld          , estimated level difference, i.e., amp. ratio.

soundVelocity=343; % used to compute the cut-off freq
sigLength = size(micSig,1); % length of the signal
frameLag = (1-overlap)*frameLength; % lag of each frame
fftLength = frameLength;  % fft point, 2^N
frameNumber = floor((sigLength-overlap*frameLength)/frameLag); % number of frames to be processed

freqStart = 300; % remove the lower freq components to reduce the effect due to reverberation
freqEnd = soundVelocity/(2*micDist)-300; % this is the highest frequency that will not introduce a phase wraparound based on mic spacing
freqResolution = fs/fftLength; 
freqStartIdx = ceil(freqStart/freqResolution);
freqEndIdx = floor(freqEnd/freqResolution);
validFreqIdx = repmat((freqStartIdx:1:freqEndIdx)',1,frameNumber); % freqs without any ambituity
ambigFreqIdx = repmat((freqEndIdx+1:1:fftLength/2)',1,frameNumber); % ambiguous freqs, i.e., > freqEnd
validFreqMat = 2*pi*freqResolution.*validFreqIdx; % 2*pi*omega, to convert the phase to the delay
ambigFreqMat = 2*pi*freqResolution.*ambigFreqIdx; % 2*pi*omega, to convert the phase to the delay
fftFreqMat = repmat(2*pi.*(1:1:fftLength/2)'*freqResolution,1,frameNumber); % complete fft freqs

% fetch the signal and implement fft
for k=1:frameNumber
    frameStart = (k-1)*frameLag+1; frameEnd = frameStart+frameLength-1;
    frameSig = micSig(frameStart:frameEnd,:);
    sig1fft(:,k) = fft(frameSig(:,1).*hamming(frameLength));%
    sig2fft(:,k) = fft(frameSig(:,2).*hamming(frameLength));%
end;

% compute the ratio, S1/S2
ratioFull = sig1fft(1:fftLength/2,:)./(sig2fft(1:fftLength/2,:)+eps);
phiFull = imag(log(ratioFull)); % get the phase term
tauFull = phiFull./fftFreqMat;  % tau is the delay between the two tf bins
ampFull = abs(ratioFull);             % amp is the attenuation ratio between the two tf bins 

% compute the ratio over the valid freq range
ratio = ratioFull(validFreqIdx);
phi = imag(log(ratio));
tau = phi./validFreqMat;  % tau is the delay between the two tf bins
amp = abs(ratio);             % amp is the attenuation ratio between the two tf bins

% histogram the delay and the amplitude ratio over valid freqs
edges{1} = (-1:0.2:1).*micDist/soundVelocity;
edges{2} = (0:0.2:5);
[hValid,cValid]=hist3([tau(:)'; amp(:)']','Edges',edges);
[binNumValid,peakIdxValid] = max(hValid(:)'); % identify the peak
[tauIdx,ampIdx] = ind2sub(size(hValid),peakIdxValid);
tdValid = cValid{1}(tauIdx); ldValid = cValid{2}(ampIdx);

phiEst = tdValid.*ambigFreqMat;  % estimate the phase at the higher freq range
phiHigh = phiFull(ambigFreqIdx); % phase term at the higher freq range
cyc = -3:1:3; % ambiguity cycles to be considered
for kk=1:size(ambigFreqIdx,1)  % remove the ambituity for the phase of each tf bin
    for ii=1:frameNumber
        phiError = phiEst(kk,ii)-phiHigh(kk,ii)+2*pi.*cyc;
        [err,cycIdx] = min(abs(phiError));
        phiUnwrap(kk,ii) = phiHigh(kk,ii)-2*pi.*cyc(cycIdx);
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
figure(1)   % plot the phase term and the unwrapped phase term
plot(1:fftLength/2,mean(phiFull,2),'b-',ambigFreqIdx(:,1),mean(phiEst,2),'r--');hold on;
plot([validFreqIdx(:,1);ambigFreqIdx(:,1)],mean([phi;phiUnwrap],2),'k:'); hold off;
xlabel('freq bin'); ylabel('phase');
legend('original phase', 'predicted phase','unwrapped phase');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tauHighFreq = phiUnwrap./ambigFreqMat; % delay at higher freqs
tauFull(ambigFreqIdx) = tauHighFreq; % delay for all freqs after removing the ambiguity
% histogram the delay and amplitude ratio over all freqs
[hFull,cFull] = hist3([tauFull(:)'; ampFull(:)']','Edges',edges);
[binNumFull,peakIdxFull] = max(hFull(:)'); % identify the peak
[tauIdxFull,ampIdxFull] = ind2sub(size(hFull),peakIdxFull); % get the index for td and ld
td = cFull{1}(tauIdxFull);
ld = cFull{2}(ampIdxFull);
