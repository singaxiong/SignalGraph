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
[D, N, T] = size(X);    % D is number of freq bin, N is number of channel

if context_size == 0    % get global spatial covariance matrix
    X2 = permute(X, [2 1 3]);
    XX = outProdND(X2);
    R = mean(XX,4);
else    % get windowed spatial covariance
    if 0
        nBlock = length(1:shift:T);
        R = zeros(N,N,D,nBlock);
        for d = 1:D
            for t = 1:shift:T
                frame_selector = t-half_ctx:t+half_ctx;
                frame_selector = min(T,max(1,frame_selector));
                X2 = squeeze(X(d,:,frame_selector));
                R(:,:,d,(t-1)/shift+1) = X2*X2' / context_size;
            end
        end
    else
        X2 = permute(X, [2 1 3]);
        XX = outProdND(X2);
%         X2cell = num2cell(X2, [1]);
%         XXcell = cellfun(@(x) (reshape(x*x', N^2,1)), X2cell, 'UniformOutput', 0);
%         XX = cell2mat(XXcell);
        XX2 = reshape(XX, N^2*D, T);
        if 1
            XX2 = gpuArray(XX2);
            idx = [ones(1,half_ctx) 1:T ones(1,half_ctx)*T];
            SCM = conv2(XX2(:,idx), ones(1,context_size, class(gather(X)))/context_size, 'valid');
%             SCM = SCM(:,half_ctx+1:end-half_ctx);
        else
            fake_layer.a = XX2;
            XX3 = F_splice(fake_layer, context_size);
            XX4 = reshape(XX3,  N^2*D, context_size, T);
            SCM = mean(XX4,2);
        end
        SCM2 = reshape(SCM, N^2, D, T);
        SCM3 = reshape(SCM2, N, N, D, T);
        R = SCM3(:,:,:,1:shift:end);
    end    
end