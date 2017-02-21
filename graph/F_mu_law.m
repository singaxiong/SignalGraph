function [output] = F_mu_law(input_layer, curr_layer)
input = input_layer.a;

if isfield(curr_layer, 'mu')
    mu = curr_layer.mu;
else
    mu = 255;
end

output = sign(input) .* log(1+mu*abs(input)) ./ log(1+mu);

end
