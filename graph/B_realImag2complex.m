function grad = B_realImag2complex(future_layers, curr_layer)

future_grad = GetFutureGrad(future_layers, curr_layer);

real_grad = real(future_grad);
imag_grad = -imag(future_grad);
grad = [real_grad; imag_grad];
end