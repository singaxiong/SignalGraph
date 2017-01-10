function output = F_largerThan(input_layer, curr_layer)
input = input_layer.a;

if isfield(curr_layer, 'threshold')
    threshold = curr_layer.threshold;
else
    threshold = 0;
end
output = input > threshold;
precision = class(gather(input(1)));
if strcmpi(precision, 'single')
    output = single(output);
else
    output = double(output);
end

end
