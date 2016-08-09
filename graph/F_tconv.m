
% this function perform convolution on the temporal direction
function [output,X2] = F_tconv(input, curr_layer)

% the input contains N samples, each of DxT size
[D,T,N] = size(input);
% Each row of W is a DxP filter to be applied along the time axis. We have
% totally H such filters.
W = curr_layer.W;
b = curr_layer.b;
[H,DP] = size(W);
P = DP/D;
halfP = (P-1)/2;

% Instead of using conv2, we splice input and use matrix transform
% We first put the N samples into a big sample, separated by zeros
if strcmpi(class(input), 'gpuArray')
    input2 = gpuArray.zeros(D,T+P,N);
else
    input2 = zeros(D,T+P,N);    
end
input2(:,halfP+1:halfP+T,:) = input;
X = reshape(input2, D, (T+P)*N);
context = -halfP : halfP;
X2 = ExpandContext_v2(X, context);

% then just perform convolution like affine transform
fakeLayer.a = X2;
Y2 = F_affine_transform(fakeLayer, W, b);

% now divide the output back to samples
Y = reshape(Y2, H, T+P, N);

if isfield(curr_layer, 'range') && strcmpi(curr_layer.range, 'valid') == 0
    output = Y(:,halfP+1:halfP+T,:);
else
    output = Y(:,halfP*2+1:T,:);    
end

end