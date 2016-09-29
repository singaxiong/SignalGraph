% take the covariance matrix of input trajectories

function curr_layer = F_ll_gmm(input, curr_layer)
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

if isfield(curr_layer, 'gconst');    gconst = curr_layer.gconst; else gconst = []; end

if isfield(curr_layer, 'post')
    post = curr_layer.post;
else
    if isfield(curr_layer, 'projection')
        [~, post] = compLikelihoodGMMFast(input, gconst, prior, mu, invCov, diagCov, curr_layer.projection);
    else
        [~, post] = compLikelihoodGMMFast(input, gconst, prior, mu, invCov, diagCov);
    end
end

% output is the EM auxiliary function

if diagCov
    if 0    % for loop version
        Z_sigma_Z = 0;
        Z_sigma_mu = 0;
        for t=1:nFr
            for j=1:nGaussian
                Z_sigma_Z = Z_sigma_Z + post(j,t) * sum( input(:,t).^2 .* invCov(:,j) );
                Z_sigma_mu = Z_sigma_mu + post(j,t) * sum( input(:,t) .* invCov(:,j) .* mu(:,j) );
            end
        end
    else    % vectorized version
        Z_sigma_Z = sum(sum(   invCov'      * input.^2 .* post ));
        Z_sigma_mu = sum(sum( (invCov.*mu)' * input    .* post ));
    end
else
    if 0    % for loop version
        Z_sigma_Z2 = 0;
        Z_sigma_mu2 = 0;
        for t=1:nFr
            for j=1:nGaussian
                Z_sigma_Z2 = Z_sigma_Z2 + post(j,t) * input(:,t)' * invCov(:,:,j) * input(:,t);
                Z_sigma_mu2 = Z_sigma_mu2 + post(j,t) * input(:,t)' * invCov(:,:,j) * mu(:,j);
            end
        end        
    end
    
    if isfield(curr_layer, 'projection')
        projection = curr_layer.projection;
    else
        if IsInGPU(invCov); projection = gpuArray.zeros(dim*nGaussian, dim); else projection = zeros(dim*nGaussian, dim); end
        for i=1:nGaussian
            projection((i-1)*dim+1:i*dim,:) = chol(invCov(:,:,i));
        end
    end
    
    data_proj = projection * input;
    data_proj = reshape(data_proj, dim, nGaussian, nFr);
    Z_sigma_Z =  sum(sum( squeeze(sum(data_proj.^2)) .* post ));
    
    mu_proj = projection * mu;
    mu_proj = reshape(mu_proj, dim, nGaussian, nGaussian);
    for i=1:nGaussian
        mu_proj2(:,i) = mu_proj(:,i,i);
    end
    
    Z_sigma_mu = sum(sum( squeeze(sum(bsxfun(@times, data_proj, mu_proj2))) .* post ));
end

% GaussianOccupation = sum(post,2)';
% mu_sigma_mu = sum( GaussianOccupation .* sum( mu.^2 .* invCov ) );
% norm_term   = sum( GaussianOccupation .* sum(log(invCov)) ) - dim * log(2*pi)*nFr;
output = Z_sigma_Z/2 - Z_sigma_mu;% + mu_sigma_mu/2 + norm_term/2;
curr_layer.a = output / nFr;   % we actually minimize negative likelihood
curr_layer.post = post;

end
