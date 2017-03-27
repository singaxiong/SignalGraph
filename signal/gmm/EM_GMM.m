% Train GMM using Expectation Maximization
% Inputs:
%   X, a MxN matrix, where M is the number of samples and N is the dimension
%   N_class, number of classes
%   fullCov, set to 1 if use full covariance matrix
%
function [prior, mu, cvr, posterior, hist_ll] = EM_GMM(X, N_class, fullCov, initModel)
[nSample, dim] = size(X);

vFloor = var(X)'/100;   % variance floor is the 10% of the global variance
if fullCov
    tmp = -ones(dim,dim)*10^10;
    for i=1:dim
        tmp(i,i) = vFloor(i);   % the diagonal elements have floor
    end
    vFloor = tmp;
end

if nargin<4
    fprintf('Initialize the class centers using K-means\n');
    Xs = X;
    [model, IDX] = my_kmeans(MVN(Xs)',N_class);
    for i=1:N_class
        idx = find(IDX == i);
        prior(i) = length(idx) / length(IDX);
        mu(:,i) = mean(Xs(idx,:));
        if fullCov
            cvr(:,:,i) = max(vFloor, cov(Xs(idx,:)));
        else
            cvr(:,i) = max(vFloor, var(Xs(idx,:))');
        end
    end
    clear Xs;
else
    prior = initModel.prior;
    mu = initModel.mu;
    cvr = initModel.cvr;
end

hist_mu(:,:,1) = mu;
if fullCov
    hist_cvr(:,:,:,1) = cvr;
else
    hist_cvr(:,:,1) = cvr;
end

% Iterative EM
fprintf('Refine model parameters using EM\n');
for itr = 1:10
    % E-step, find the posterior probabiities of the Gaussians given the data
    if 0
        for i=1:N_class
            if fullCov
                likelihood(:,i) = prior(i) * mvnpdf(X, mu(:,i)', cvr(:,:,i));
            else
                likelihood(:,i) = prior(i) * my_mvnpdf(X, mu(:,i)', cvr(:,i)',1);
            end
        end
        evidence = sum(likelihood');
        posterior = likelihood ./ repmat(evidence', 1,N_class);
        hist_ll(itr)= sum(log(evidence))/nSample;
    else
        if fullCov
            for i=1:N_class
                invcvr(:,:,i) = inv(cvr(:,:,i));
            end
        else
            invcvr = 1./cvr;
        end
        
        [~, posterior, hist_ll(itr)] = compLikelihoodGMMFast(X', [], prior, mu, invcvr, fullCov==0);
        posterior = posterior';
    end

    fprintf('Iteration %d, average log likelihood per frame = %f - %s\n', itr, hist_ll(itr), datestr(now));
    
    % M-step
    for i=1:N_class
        [mu(:,i),tmp] = findMeanVarainceWeighted(X, posterior(:,i),fullCov);
        prior(i) = sum(posterior(:,i)) / nSample;
        if fullCov
            cvr(:,:,i) = max(vFloor, tmp);
        else
            cvr(:,i) = max(vFloor, tmp);
        end
        %fprintf('%d elements floored\n', sum(tmp<vFloor));
    end
    
    hist_prior(:,itr+1) = prior;
    hist_mu(:,:,itr+1) = mu;
    if fullCov
        hist_cvr(:,:,:,itr+1) = cvr;
    else
        hist_cvr(:,:,itr+1) = cvr;
    end
    
    if(itr>1)   % stopping criterion
        if abs(hist_ll(itr)-hist_ll(itr-1)) / abs(hist_ll(itr-1)) < 0.0001
            break;
        end
    end
end

% Even if we cluster data using diagonal covariance, we still output full
% covariance
if fullCov==0
    cvr = zeros(dim,dim,N_class);
    for i=1:N_class
        [tmp,cvr(:,:,i)] = findMeanVarainceWeighted(X, posterior(:,i),1);
        for j=1:dim
            cvr(j,j,i) = max( cvr(j,j,i), vFloor(j));
        end
    end    
end

if fullCov
    for i=1:itr+1
        for j=1:N_class
            hist_var(:,j,i) = diag(hist_cvr(:,:,j,i));
        end
    end
end

