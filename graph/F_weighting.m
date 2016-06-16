function output = F_weighting(input, weight, bias)
[dim, N] = size(input);
nWeight = length(weight);
D = dim/nWeight;

input2 = reshape(input, D, nWeight, N);
input2 = permute(input2, [2 1 3]);

output = sum(bsxfun(@times, input2, weight));

if N==1
    output = output';
else
    output = permute(output, [2 3 1]);
end

output = bsxfun(@plus, output, bias);

end
