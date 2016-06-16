%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%@function: Calculate the correlation vectors for all the permutations of the given input channels.
%%%The method uses inverse Fourier transform of the cross-power spectrum to
%%%obtain the cross-correlation based on GCC-PHAT (C Knapp, 1976).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correlationVectors]=getCorrelationVector_fast(input, winSize, overlap)

%%input:   input signals with dimesion M x N, where M is the number of
%%%        channels, and N is the total length of signal
%%winSize: the window size used for computing the correlation vector
%%overlap: the percentage of overlap between windows

%%correlationVectors: output of the correlation vectors with the size of
%%winSize x (M*(M-1)/2) x the number of windows in input signal

M = size(input,1);%the number of channels in the array
N = size(input,2);%%total length of input signal
window = hamming(winSize);

nWindowFrames = floor((N-winSize)/(winSize*(1-overlap))+1); %%compute the total window frames in the input signals
if nWindowFrames ==0 && N > winSize/3
    nWindowFrames = 1;
end
    
nCorreVect = M*(M-1)/2;    %%compute the number of permutations for the correlation vectors given M channels
%%for example, if M=3, the correlations are (1,2), (1,3), and (2,3)
correlationVectors = zeros(winSize,nWindowFrames,nCorreVect); %%output of the correlation vectors
Pxy = correlationVectors;

window_Segs = zeros(winSize,M); %%window buffer to store M signal segments
for iframe=1:nWindowFrames
    if iframe==1
        st = 1;
        et = min(N,winSize);
        window_Segs = input(:,st:et)';
        if length(window_Segs)<winSize
            window_Segs(size(window_Segs,1)+1:winSize,:) = 0;
        end
    else
        st = (iframe-1)*(winSize*(1-overlap))+1;
        et = min(N,iframe*(winSize*(1-overlap)));
        buff = input(:,st:et)';
        window_Segs = [window_Segs(size(buff,1)+1:end,:);buff];
    end
   
    count = 1;
    allfft = fft(bsxfun(@times,window_Segs,window));
    for m=1:M-1
        fftx=allfft(:,m);
        for l=m+1:M
            ffty=allfft(:,l);
            Pxy(:,iframe,count)=fftx.*conj(ffty);  % compute the cross-spectrum
            count = count + 1;
        end
    end
end

for i=1:size(Pxy,3)
    currPxy = Pxy(:,:,i);
    weight_factor=max(abs(currPxy),1e-6); %create the weight factor
    norm_Pxy=currPxy./weight_factor;
    r_GCC=fftshift(real(ifft(norm_Pxy)),1);
    r_GCC=flipud(r_GCC);
    correlationVectors(:,:,i) = r_GCC;
end


