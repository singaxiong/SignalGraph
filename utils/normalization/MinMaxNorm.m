%
% Perform minmax norm
%
function [output, input_min, input_max, scale] = MinMaxNorm(input,min_val, max_val)
if nargin==1
    min_val = -1;
    max_val = 1;
end

input_min = min(input(:));
input_max = max(input(:));

input_range = input_max-input_min;
target_range = max_val - min_val;
scale = target_range ./ input_range;

output = bsxfun(@plus, input, -input_min);
output = bsxfun(@times, output, scale);
output = bsxfun(@plus, output, min_val);

end
