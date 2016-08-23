function output = F_median(input_layer, curr_layer)
input = input_layer.a;
if isfield(curr_layer, 'idx')
    output = median(input, curr_layer.idx);
else
    output = median(input,2);
end
end

