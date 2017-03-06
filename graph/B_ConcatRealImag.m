function grad = B_ConcatRealImag(prev_layer, curr_layer, future_layers)

future_grad = GetFutureGrad(future_layers, curr_layer);

[D, T, N] = size(future_grad);
j = sqrt(-1);

realpart = future_grad(1:D/2,:,:);
imagpart = future_grad(D/2+1:end,:,:);

grad = realpart + j*imagpart;

end