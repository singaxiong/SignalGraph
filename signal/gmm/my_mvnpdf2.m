function prob = my_mvnpdf2(x,mu,invCov)
diagCov = size(invCov,1) ~= size(invCov,2);
[dim, nFr] = size(x);

if diagCov == 0
    prob = mvnpdf(x',mu,inv(invCov));
else
    norm_term = ( (2*pi)^(dim/2) ) / prod( sqrt(invCov) );
    tmp = bsxfun(@minus, x, mu(:));
    tmp = tmp.*tmp;
    tmp = bsxfun(@times, tmp, invCov(:));
    tmp = sum(tmp,1);
    prob = exp(-tmp/2) / norm_term;
end

end