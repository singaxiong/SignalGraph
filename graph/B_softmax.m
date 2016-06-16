function [grad] = B_softmax(output)
grad = output .* (1-output);
end
