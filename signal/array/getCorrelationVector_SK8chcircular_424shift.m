%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%@function: Calculate the correlation vectors for all the permutations of the given input channels.
%%%The method uses inverse Fourier transform of the cross-power spectrum to
%%%obtain the cross-correlation based on GCC-PHAT (C Knapp, 1976).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correlationVectors]=getCorrelationVector_SK8chcircular_424shift(input, winSize, overlap)

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
nCorreVect = M*(M-1)/2;    %%compute the number of permutations for the correlation vectors given M channels
                           %%for example, if M=3, the correlations are (1,2), (1,3), and (2,3)
correlationVectors = zeros(winSize,nWindowFrames,nCorreVect); %%output of the correlation vectors
window_Segs = zeros(winSize,M); %%window buffer to store M signal segments
for iframe=1:nWindowFrames    
    if iframe==1
        st = 1;
        et = winSize;
        window_Segs = input(:,st:et)';
    else
        st = (iframe-1)*(winSize*(1-overlap))+1;
        et = iframe*(winSize*(1-overlap));
        buff = input(:,st:et)';
        window_Segs = [window_Segs(size(buff,1)+1:end,:);buff];
    end
    count = 1;
    for m=1:M-1
        for l=m+1:M
            fftx=fft(window_Segs(:,m).*window);
            ffty=fft(window_Segs(:,l).*window);
            Pxy=fftx.*conj(ffty);  % compute the cross-spectrum
            weight_factor=max(abs(Pxy),1e-6); %create the weight factor
            norm_Pxy=Pxy./weight_factor;
            r_GCC=fftshift(real(ifft(norm_Pxy)),1);
            r_GCC=flipud(r_GCC);
            
            % to counter the effect of wrong synchronization between first
            % 4 and last 4 channels. 
            if m<5 && l>4
                r_GCC = circshift(r_GCC,-154);
            end

%             figure(1);plot([-winSize/2+1:winSize/2],r_GCC,'r');pause(0.1);hold off;
            correlationVectors(:,iframe,count) = r_GCC;
            count = count + 1;
        end
    end
end

