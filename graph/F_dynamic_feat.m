function [output,validFrameMask] = F_dynamic_feat(input_layer)
input = input_layer.a;
[D, T, N] = size(input);

if N>1
    [validFrameMask, variableLength] = getValidFrameMask(input_layer);
    if variableLength; input = PadShortTrajectory(input, validFrameMask, 'last'); end
else
    if 0    % implementing using linear transform
        Dmat = genDeltaTransform(T, 2);
        Amat = Dmat*Dmat;
        output = [input; input*Dmat'; input*Amat'];
    else
        output = comp_dynamic_feature(input',2,2)';
    end
    validFrameMask = [];
    return;
end

if 0    % note that due to the taking of double delta, the N=1 and N>1 cases are not exactly the same at the boundaries in this version.
    context = 2*4+1;
    halfC = (context-1)/2;
    input2 = Tensor2MatWithBuffer(input, context, 1);
    output2 = comp_dynamic_feature(input2',2,2)';
    output2 = reshape(output2, size(output2,1), T+context, N);
    output = output2(:,halfC+1:halfC+T,:);
else    % this version produces exact results
    if variableLength
        input2 = ExtractVariableLengthTrajectory(input, validFrameMask);
        if IsInGPU(input)
            output = gpuArray.zeros(D*3,T,N);
        else
            output = zeros(D*3,T,N);
        end
        for i=1:N
            output(:,1:size(input2{i},2),i) = comp_dynamic_feature(input2{i}',2,2)';
        end
    else
        input2 = reshape(permute(input, [1 3 2]), D*N,T);
        output2 = comp_dynamic_feature(input2',2,2)';
        output2 = reshape(output2, D, N, 3, T);
        output = reshape(permute(output2, [1 3 4 2]), D*3,T,N);
    end
end
 
%  if 0
%     output3 = zeros(size(output));
%     for i=1:N
%         output3(:,:,i) = comp_dynamic_feature(input(:,:,i)',2,2)';
%     end
% end
end
