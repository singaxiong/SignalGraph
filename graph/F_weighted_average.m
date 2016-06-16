function [output, weights] = F_weighted_average(prev_layers)

raw_weights = prev_layers{1}.a;
input = prev_layers{2}.a;
[dim, nFr] = size(input);

raw_weights2 = exp(raw_weights);
weights = raw_weights2 / sum(raw_weights2);

output = input * weights';

%output = repmat(output, 1, nFr);

end
