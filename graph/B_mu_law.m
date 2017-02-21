function [grad] = B_mu_law(future_layers, curr_layer)
input = input_layer.a;

future_grad = GetFutureGrad(future_layers, curr_layer);

if isfield(curr_layer, 'mu')
    mu = curr_layer.mu;
else
    mu = 255;
end

% output = sign(input) .* log(1+mu*abs(input)) ./ log(1+mu);
grad = future_grad .* sign(input) ./ (1+mu*abs(input)) * (mu/log(1+mu));

end
