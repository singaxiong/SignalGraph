function [input2, weight, input_layer] =  prepareBeamforming(input_layers)
for i=1:length(input_layers)
    if strcmpi(input_layers{i}.name, 'real_imag2BFweight') || strcmpi(input_layers{i}.name, 'MVDR_spatialCov')
        N = length(input_layers{i}.freqBin);
        weight = input_layers{i}.a;
        if isfield(input_layers{i}, 'online')
            online = input_layers{i}.online;
        else
            online = 0;
        end
    else
        input = input_layers{i}.a;
        input_layer = input_layers{i};
    end
end
[NC,T,nSent] = size(weight);
[NC,T,nSent] = size(input);
C = NC/N;
if online
    weight = reshape(weight, N,C,T,nSent);
else
    weight = reshape(weight, N,C,1,nSent);
end
input2 = reshape(input, N,C,T,nSent);


