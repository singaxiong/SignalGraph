% take the covariance matrix of input trajectories

function curr_layer = F_ll_gmm(input, curr_layer)
useGPU = strcmpi(class(input(1)), 'gpuArray');

prior = curr_layer.prior;
mu = curr_layer.mu;
invCov = curr_layer.invCov;

[dim,nFr] = size(input);
nMix = length(prior);
[d1,d2,d3] = size(invCov);
if nMix==1
    if d2==1; diagCov = 1; else diagCov = 0; end
else
    if d3==1; diagCov = 1; else diagCov = 0; end
end

if isfield(curr_layer, 'gconst');    gconst = curr_layer.gconst; else gconst = []; end

if isfield(curr_layer, 'post')
    post = curr_layer.post;
else
    [~, post] = compLikelihoodGMMFast(input, gconst, prior, mu, invCov, diagCov, useGPU);
end

% output is the EM auxiliary function

if 0    % for loop version
    Z_sigma_Z = 0;
    Z_sigma_mu = 0;
    for t=1:nFr
        for j=1:nMix
            Z_sigma_Z = Z_sigma_Z + post(j,t) * sum( input(:,t).^2 .* invCov(:,j) );
            Z_sigma_mu = Z_sigma_mu + post(j,t) * sum( input(:,t) .* invCov(:,j) .* mu(:,j) );
        end
    end
else    % vectorized version
    Z_sigma_Z = sum(sum(   invCov'      * input.^2 .* post ));
    Z_sigma_mu = sum(sum( (invCov.*mu)' * input    .* post ));
end

% GaussianOccupation = sum(post,2)';
% mu_sigma_mu = sum( GaussianOccupation .* sum( mu.^2 .* invCov ) );
% norm_term   = sum( GaussianOccupation .* sum(log(invCov)) ) - dim * log(2*pi)*nFr;
output = Z_sigma_Z/2 - Z_sigma_mu;% + mu_sigma_mu/2 + norm_term/2;
curr_layer.a = output / nFr;   % we actually minimize negative likelihood
curr_layer.post = post;

end
