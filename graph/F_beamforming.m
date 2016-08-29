
function [output] = F_beamforming(input_layers, curr_layer)
weight = input_layers{1}.a;
nBin = length(curr_layer.freqBin);
input = input_layers{2}.a;

[Di,Ti,Ni] = size(input);

output = bsxfun(@times, input, conj(weight));

output = reshape(output, nBin,Di/nBin,Ti,Ni);

output = squeeze(sum(output,2));

        
end
