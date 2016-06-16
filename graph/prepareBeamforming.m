function [input2, weight] =  prepareBeamforming(input_layers)
for i=1:length(input_layers)
    if strcmpi(input_layers{i}.name, 'input')
        input = input_layers{i}.a;
    else
        N = length(input_layers{i}.freqBin);
        weight = input_layers{i}.a;
        online = input_layers{i}.online;
    end
end
[NC,T] = size(weight);
[NC,T] = size(input);
C = NC/N;
if online
    weight = reshape(weight, N,C,T);
end
input2 = reshape(input, N,C,T);


