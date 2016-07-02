%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%@function: Calculate the correlation vectors for all the permutations of the given input channels.
%%%The method uses inverse Fourier transform of the cross-power spectrum to
%%%obtain the cross-correlation based on GCC-PHAT (C Knapp, 1976).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correlationVectors]=getCorrelationVector_fast2(input, winSize, overlap, useGPU)
if nargin<4
    useGPU = 0;
end
%%input:   input signals with dimesion M x N, where M is the number of
%%%        channels, and N is the total length of signal
%%winSize: the window size used for computing the correlation vector
%%overlap: the percentage of overlap between windows

%%correlationVectors: output of the correlation vectors with the size of
%%winSize x (M*(M-1)/2) x the number of windows in input signal

M = size(input,1);%the number of channels in the array
N = size(input,2);%%total length of input signal
window = hamming(winSize);

winShift = round(winSize * (1-overlap));
X = my_enframe(input', winSize, winShift, 1-overlap/2);
nWindow = size(X,3);
if nWindow ==0
    correlationVectors = [];
    return;
end

nCorreVect = M*(M-1)/2;    %%compute the number of permutations for the correlation vectors given M channels
%%for example, if M=3, the correlations are (1,2), (1,3), and (2,3)

Xfft = fft(bsxfun(@times,X,window));
Xfft = permute(Xfft, [1 3 2]);

if useGPU    % fully vectorized version is much faster for GPU
    idx = [];
    for m=1:M-1
        for l=m+1:M
            idx(1,end+1) = m;
            idx(2,end) = l;
        end
    end
    Pxy = Xfft(:,:,idx(1,:)) .* conj( Xfft(:,:,idx(2,:)) );
else
    Pxy = zeros(winSize,nWindow,nCorreVect);
    count = 1;
    for m=1:M-1
        fftx=Xfft(:,:,m);
        for l=m+1:M
            ffty=Xfft(:,:,l);
            Pxy(:,:,count)=fftx.*conj(ffty);  % compute the cross-spectrum
            count = count + 1;
        end
    end
end

Pxy = reshape(Pxy, winSize, nCorreVect*nWindow);
weight_factor=max(abs(Pxy),1e-6); %create the weight factor
norm_Pxy=Pxy./weight_factor;
r_GCC=fftshift(real(ifft(norm_Pxy)),1);
r_GCC=flipud(r_GCC);
correlationVectors = r_GCC;
correlationVectors = reshape(correlationVectors, winSize,nWindow,nCorreVect);

end
