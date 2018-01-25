function [output] = softmax(input)
output = exp(input);
recip_sum_a = 1./sum(output);
output = bsxfun(@times, output, recip_sum_a);

end
