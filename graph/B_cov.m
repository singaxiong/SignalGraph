function grad = B_cov(input, future_layers)

[dim, nFr] = size(input);

future_grad = GetFutureGrad(future_layers, {});

grad = 2 * future_grad * input / nFr;

end