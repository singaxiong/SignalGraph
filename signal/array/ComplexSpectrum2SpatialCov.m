% this function compute the spatial covariance matrix from complex Fourier
% transform of array signals.
% Inputs:
%   X: D x N x T matrix of Fourier transform coefficients. D is the number
%   of frequency bins, N is the number of microphone channels, and T is the
%   number of frames
%   context_size: number of context frames used to compute the spatial
%   covariance at each time-frequency bin.
% Outputs:
%   R: N x N x D x T matrix of spatial covariance matrixes
%
% Author: Xiong Xiao, Nanyang Technological University, Singapore
% Last Modified: 22 Feb 2016.
%
function R = ComplexSpectrum2SpatialCov(X, context_size, shift)

half_ctx = (context_size-1)/2;
[D, N, T] = size(X);

if context_size == 0    % get global spatial covariance matrix
    R = zeros(N,N,D);
    for d = 1:D
        X2 = squeeze(X(d,:,:));
        R(:,:,d) = X2*X2' / T;
    end
else    % get windowed spatial covariance
    nBlock = length(1:shift:T);
    R = zeros(N,N,D,nBlock);
    for d = 1:D
        %     tic
        for t = 1:shift:T
            frame_selector = t-half_ctx:t+half_ctx;
            frame_selector = min(T,max(1,frame_selector));
            X2 = squeeze(X(d,:,frame_selector));
            R(:,:,d,(t-1)/shift+1) = X2*X2' / context_size;
        end
        %     toc
        %     tic
        %     a = squeeze(X(d,:,:));
        %     cellmatr = arrayfun(@(x) a(:,x) * a(:,x).', 1:size(a,2), 'uni', 0);
        %     toc
    end
    
end