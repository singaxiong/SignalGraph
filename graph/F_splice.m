function output = F_splice(input, context)
[D, T, N] = size(input);
halfC = (context-1)/2;
if context==1
    output = input;
end

if N==1
    output = ExpandContext_v2(input, -halfC : halfC);
    return;
end

input2 = Tensor2MatWithBuffer(input, context, 1);
output2 = ExpandContext_v2(input2, -halfC : halfC);
output2 = reshape(output2, size(output2,1), T+context, N);
output = output2(:,halfC+1:halfC+T,:);

if 0
    output3 = zeros(size(output));
    for i=1:N
        output3(:,:,i) = ExpandContext_v2(input(:,:,i), -halfC : halfC);
    end
end
end