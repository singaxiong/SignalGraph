function grad = B_real_imag2BFweight(future_layers, curr_layer, T)

future_grad = GetFutureGrad(future_layers, curr_layer);
grad = [real(future_grad); -imag(future_grad)];
grad = repmat(grad, 1, T)/T;
end
