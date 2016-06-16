function [cost, grad, cost_pure, negdata, energy] = RBM_cost(data, rbm, para, mode)
if nargin<4
    mode = 1;
end

numhid = length(rbm.hidbiases);
numcases = size(data,1);
vishid = rbm.vishid;
hidbiases = rbm.hidbiases;
visbiases = rbm.visbiases;
numdims = para.layerSize(1);
var_data = ones(1,numdims);

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


poshidprobs = 1./(1 + exp(-bsxfun(@plus, data*vishid, hidbiases)));
%poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    % posterior prob. that the hidden unit is on
if mode<2
    posprods    = data' * poshidprobs;              % gradient w.r.t. W summed over 100 utterances
    poshidact   = sum(poshidprobs);                 % gradient w.r.t. hidden units' biases summed over 100 utterances
    posvisact = sum(data);                          % gradient w.r.t. visible units' biases summed over 100 utterances
end

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(para, 'MeanFieldApprox') && para.MeanFieldApprox==1
    poshidstates = poshidprobs;
else
    poshidstates = single(poshidprobs > rand(numcases,numhid));                         % Sample from the posterior prob. of the hidden units
end

if mode==2
    term1 = -rbm.hidbiases*poshidstates';
    term2 = sum(bsxfun(@plus, data, -rbm.visbiases).^2,2)';
    term3 = -diag(data*rbm.vishid*poshidstates')';
    energy = term1 + term2 + term3;
end

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmpi(para.InputPDF, 'Gaussian')  % use Gaussian distribution for the inputs
    % mu_data = repmat(visbiases, numcases, 1) + double(poshidstates) * vishid' .* repmat((var_data), numcases,1);
    mu_data = bsxfun(@plus, single(poshidstates) * vishid', visbiases); 
    if 0
        negdata = randn(size(mu_data));
        negdata = negdata .* repmat(sqrt(var_data), numcases,1);
        negdata = negdata + mu_data;
    else
        negdata = mu_data;
    end
else
    negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));   % Sample visible data given the sample of the hidden units
end

if mode==2
    grad = []; cost = 0; cost_pure = 0;
    return;
end

% neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));     % Posterior of the hidden units given the sampled visibles
neghidprobs = 1./(1 + exp(-bsxfun(@plus, negdata*vishid, hidbiases)));     % Posterior of the hidden units given the sampled visibles
negprods  = negdata'*neghidprobs;           % gradient w.r.t. W summed over 100 utterances, but using sampled data
neghidact = sum(neghidprobs);               % gradient w.r.t. hidden units' biases summed over 100 utterances, but using sampled data
negvisact = sum(negdata);                   % gradient w.r.t. visible units' biases summed over 100 utterances, but using sampled data

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cost = 0.5*sum(sum( (data-negdata).^2 ))/numcases;         % Mean square error between true data and sampled visible data
cost_pure = cost;

if isinf(cost) || isnan(cost)
    pause(0.1);
end

grad.vishid = (posprods-negprods)/numcases;
grad.visbiases = (posvisact-negvisact)/numcases;
grad.hidbiases = (poshidact-neghidact)/numcases;
if para.L2weight>0
    grad.vishid = grad.vishid - para.L2weight*rbm.vishid; 
    cost = cost + 0.5* para.L2weight * sum(sum(rbm.vishid.*rbm.vishid));
end

if max(max(abs(grad.vishid)))>10
    pause(0.1);
end
