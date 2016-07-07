function prob = my_mvnpdf(x,meanV,varV,diagCov)
if nargin < 3
    diagCov = 0;
end
if diagCov == 0
    prob = mvnpdf(x,meanV,varV);
else
    % meanV, varV are row vectors
    % each row of x is a sample
    [N_vector,Dim] = size(x);
    norm_term = ( (2*pi)^(Dim/2) ) * prod( sqrt(varV) );
    if 1
        tmp = bsxfun(@minus, x, meanV);
        tmp = tmp.*tmp;
        tmp = bsxfun(@times, tmp, 1./varV);
    end
    if 0
        tmp = (x - repmat(meanV,N_vector,1)).^2;
        tmp = tmp ./ repmat(varV,N_vector,1);
    end
    tmp = sum(tmp,2);
    prob = exp(-tmp/2) / norm_term;
end

end