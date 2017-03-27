function grad = B_ll_gaussian(prev_layers, curr_layer)

mu = prev_layers{1}.a;
variance = prev_layers{2}.a;
input = prev_layers{3}.a;

[D,T,N] = size(input);
if N==1
    nFrame = T;
else
    [validFrameMask, variableLength] = getValidFrameMask(prev_layers{3});
    nFrame = N*T - sum(validFrameMask(:));
end

tmp = (mu-input) ./ variance;
gradX = tmp / nFrame;
gradMu = -gradX;

gradVar = tmp.^2 - 1./variance;
gradVar = gradVar / (2*nFrame);

grad{1} = -gradMu;
grad{2} = -gradVar;
grad{3} = -gradX;

end