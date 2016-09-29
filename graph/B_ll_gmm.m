function grad = B_ll_gmm(input, curr_layer)
useGPU = strcmpi(class(input(1)), 'gpuArray');

prior = curr_layer.prior;
mu = curr_layer.mu;
invCov = curr_layer.invCov;

[dim,nFr] = size(input);
nGaussian = length(prior);
[d1,d2,d3] = size(invCov);
if nGaussian==1
    if d2==1; diagCov = 1; else diagCov = 0; end
else
    if d3==1; diagCov = 1; else diagCov = 0; end
end

post = curr_layer.post;

if diagCov
    if 0   % for loop version
        if useGPU
            z_sigma = gpuArray.zeros(size(input));
        else
            z_sigma = zeros(size(input));
        end
        mu_sigma = z_sigma;
        for t=1:nFr
            for j=1:nGaussian
                z_sigma(:,t) = z_sigma(:,t) + post(j,t) * input(:,t) .* invCov(:,j);
                mu_sigma(:,t) = mu_sigma(:,t) + post(j,t) * mu(:,j) .* invCov(:,j);
            end
        end
    else    % vectorized version
        z_sigma = bsxfun(@times, input, reshape(invCov,dim,1,nGaussian));
        z_sigma = bsxfun(@times, z_sigma, reshape(post', 1, nFr,nGaussian));
%         z_sigma2 = bsxfun(@times, input, permute(invCov,[1 3 2]));
%         z_sigma2 = bsxfun(@times, z_sigma2, permute(post', [3 1 2]));
        
        z_sigma = squeeze(sum(z_sigma,3));
        mu_sigma = (mu.*invCov) * post;
    end
else    
    if 0    % for loop version
        if useGPU
            z_sigma2 = gpuArray.zeros(size(input));
        else
            z_sigma2 = zeros(size(input));
        end
        mu_sigma2 = z_sigma2;
        for t=1:nFr
            for j=1:nGaussian
                z_sigma2(:,t) = z_sigma2(:,t) + post(j,t) * invCov(:,:,j) * input(:,t);
                mu_sigma2(:,t) = mu_sigma2(:,t) + post(j,t) * invCov(:,:,j) * mu(:,j);
            end
        end        
    end

    projection = reshape(invCov, dim, dim*nGaussian)';
    data_proj = projection * input;
    data_proj = reshape(data_proj, dim, nGaussian, nFr);
    
    z_sigma =  bsxfun(@times, data_proj, permute(post, [3 1 2]));
    z_sigma = squeeze(sum(z_sigma,2));

    mu_proj = projection * mu;
    mu_proj = reshape(mu_proj, dim, nGaussian, nGaussian);
    for i=1:nGaussian
        mu_proj2(:,i) = mu_proj(:,i,i);
    end
    mu_sigma = mu_proj2 * post;
end

grad = -mu_sigma + z_sigma;
grad = grad / nFr;

end