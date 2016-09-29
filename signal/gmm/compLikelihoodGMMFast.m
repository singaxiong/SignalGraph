% compute the posterior and likelihood of data on a GMM
function [likelihood, posterior, avgLL] = compLikelihoodGMMFast(data, gconst, prior, mu, invCov, diagCov, projection)
nGaussian = length(prior);
[dim,nFr] = size(data);

if 0
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
else        % vectorized
    if diagCov
        %norm_term = log(prior) + sum(log(invCov))/2 - dim/2 * log(2*pi);
        if ~isempty(gconst)
            norm_term = gconst;
        else
            norm_term = log(gather(prior)) + sum(log(gather(invCov)))/2 - dim/2 * log(2*pi);
        end
        o_sigma_o = invCov' * (data.*data);
        mu_sigma_mu = sum(invCov .* (mu.*mu))';
        o_sigma_mu = (mu.*invCov)' * data;
        
        logLL = bsxfun(@plus, (-o_sigma_o/2 + o_sigma_mu), -mu_sigma_mu/2+norm_term');
        likelihood = exp(logLL);
    else
        if ~isempty(gconst)
            norm_term = gconst;
        else
            norm_term = log(prior)- dim/2 * log(2*pi);
            for i=1:nGaussian
                norm_term(i) = norm_term(i) + log(det( invCov(:,:,i) ))/2;
            end
        end
        
        if ~exist('projection', 'var')
            if IsInGPU(invCov); projection = gpuArray.zeros(dim*nGaussian, dim); else projection = zeros(dim*nGaussian, dim); end
            for i=1:nGaussian
                projection((i-1)*dim+1:i*dim,:) = chol(invCov(:,:,i));
            end
        end
        
        data_proj = projection * data;
        data_proj = reshape(data_proj, dim, nGaussian, nFr);
        mu_proj = projection * mu;
        mu_proj = reshape(mu_proj, dim, nGaussian, nGaussian);
        for i=1:nGaussian
            mu_proj2(:,i) = mu_proj(:,i,i);
        end
        
        o_sigma_o = squeeze(sum(data_proj .* data_proj));
        mu_sigma_mu = sum(mu_proj2 .* mu_proj2)';
        o_sigma_mu = squeeze(sum(bsxfun(@times, data_proj, mu_proj2)));
        
        logLL = bsxfun(@plus, (-o_sigma_o/2 + o_sigma_mu), -mu_sigma_mu/2+norm_term');
        likelihood = exp(logLL);
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
