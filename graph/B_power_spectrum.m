function grad = B_power_spectrum(ComplexSpectrum, future_layers)

future_grad = GetFutureGrad(future_layers, {});
grad = 2 * future_grad .* conj(ComplexSpectrum);      % see eq (211) of matrix cookbook 2008 version.

end
