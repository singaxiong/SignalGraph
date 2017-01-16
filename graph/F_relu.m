function output = F_relu(input_layer, curr_layer)
input = input_layer.a;

if isfield(curr_layer, 'threshold')
    threshold = curr_layer.threshold;
else
    threshold = 0;
end
output = max(threshold, input);
end
