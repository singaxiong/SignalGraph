function grad = B_ll_gmm(input, curr_layer)
useGPU = strcmpi(class(input(1)), 'gpuArray');

prior = curr_layer.prior;
mu = curr_layer.mu;
invCov = curr_layer.invCov;

[dim,nFr] = size(input);
nMix = length(prior);

post = curr_layer.post;

if 0   % for loop version
    if useGPU
        z_sigma = gpuArray.zeros(size(input));
    else
        z_sigma = zeros(size(input));
    end
    mu_sigma = z_sigma;
    for t=1:nFr
        for j=1:nMix
            z_sigma(:,t) = z_sigma(:,t) + post(j,t) * input(:,t) .* invCov(:,j);
            mu_sigma(:,t) = mu_sigma(:,t) + post(j,t) * mu(:,j) .* invCov(:,j);
        end
    end
else    % vectorized version
    z_sigma = bsxfun(@times, input, reshape(invCov,dim,1,nMix));
    z_sigma = bsxfun(@times, z_sigma, reshape(post', 1, nFr,nMix));
    z_sigma = squeeze(sum(z_sigma,3));
    mu_sigma = (mu.*invCov) * post;
end

grad = -mu_sigma + z_sigma;
grad = grad / nFr;

end