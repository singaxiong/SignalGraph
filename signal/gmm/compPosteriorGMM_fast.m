function [posterior likelihood avgLL globalPost] = compPosteriorGMM_fast(data, prior, meanC, covC, useDiag, useGPU)
if nargin<6
    useGPU = 0;
end

nClass = length(prior);
[d1 d2] = size(covC(:,:,1));
if d1 == d2
    HasFullCov = 1;
else
    HasFullCov = 0;
end
if useGPU
    meanC = gpuArray(meanC);
    covC = gpuArray(covC);
end

for j=1:nClass
    if HasFullCov==0
        likelihood(:,j) = prior(j) * my_mvnpdf(data, meanC(:,j)', covC(:,j)', 1);
    elseif useDiag
        likelihood(:,j) = prior(j) * my_mvnpdf(data, meanC(:,j)', diag(covC(:,:,j))', 1);
    else
        likelihood(:,j) = prior(j) * my_mvnpdf(data, meanC(:,j)', covC(:,:,j), 0);
    end
    
    % meanV, varV are row vectors
    % each row of x is a sample
%     [N_vector,Dim] = size(x);
%     norm_term = ( (2*pi)^(Dim/2) ) * prod( sqrt(varV) );
%     tmp = (x - repmat(meanV,N_vector,1)).^2;
%     tmp = tmp ./ repmat(varV,N_vector,1);
%     tmp = sum(tmp,2);
%     prob = exp(-tmp/2) / norm_term;
    
end
evidence = sum(likelihood');
posterior = gather( likelihood ./ repmat(evidence', 1,nClass) );
avgLL = gather( mean(log(evidence)) );

globalLL = mean(log(likelihood));
globalEvidence = sum(exp(globalLL));
globalPost = gather(exp(globalLL)/globalEvidence);
likelihood = gather(likelihood);