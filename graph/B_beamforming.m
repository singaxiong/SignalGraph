function grad = B_beamforming(future_layers, input_layers, curr_layer)
weight = input_layers{1}.a;
nBin = length(curr_layer.freqBin);
input = input_layers{2}.a;

[Di,Ti,Ni] = size(input);
[Dw,Tw,Nw] = size(weight);

future_grad = GetFutureGrad(future_layers, curr_layer);
future_grad = permute(future_grad, [1 2 4 3]);
input = reshape(input, nBin,Di/nBin,Ti,Ni);

input2 = permute(input, [1 3 2 4]);
grad{1} = bsxfun(@times, conj(input2), future_grad);
grad{1} = permute(grad{1}, [1 3 2 4]);
grad{1} = reshape(grad{1}, Di,Ti,Ni);

if Tw==1
    grad{1} = sum(grad{1},2);
end

grad{2} = [];

end
