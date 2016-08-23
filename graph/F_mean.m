function output = F_mean(input_layer, curr_layer)
input = input_layer.a;
if isfield(curr_layer, 'idx')
    output = mean(input, curr_layer.idx);
else
    output = mean(input,2);
end
end

