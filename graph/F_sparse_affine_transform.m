
function output = F_sparse_affine_transform(input, transform, bias, singlePrecision)

[D,M,N] = size(input);
if N>1
    input = reshape(input, D,M*N);
end

visible_nonzero_idx = find(sum(abs(input),2)>0);
visible_nonzero = full(input(visible_nonzero_idx,:));
if singlePrecision==1
    visible_nonzero = single(visible_nonzero); 
end
output = bsxfun(@plus, transform(:,visible_nonzero_idx) * visible_nonzero, bias);    % only compute the nonzero elements

if N>1
    output = reshape(output, size(output,1), M,N);
end

end