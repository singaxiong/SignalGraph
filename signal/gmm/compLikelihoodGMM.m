% compute the posterior and likelihood of data on a GMM
function [likelihood, posterior, avgLL] = compLikelihoodGMM(data, prior, mu, invCov, diagCov, useGPU)
nGaussian = length(prior);
[dim,nFr] = size(data);

if useGPU
    likelihood = gpuArray.zeros(nGaussian, nFr);
else
    likelihood = zeros(nGaussian, nFr);
end

for j=1:nGaussian
    if diagCov
        likelihood(j,:) = prior(j) * my_mvnpdf2(data, mu(:,j)', invCov(:,j)');
    else
        likelihood(j,:) = prior(j) * my_mvnpdf2(data, mu(:,j)', invCov(:,:,j));
    end    
end

if nargout>=2
    evidence = sum(likelihood,1);
    posterior = bsxfun(@times, likelihood, 1./evidence);
end
if nargout==3
    avgLL = mean(log(evidence));
end

end
