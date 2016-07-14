
function grad = B_dynamic_feat(curr_layer, future_layers)
output = curr_layer.a;
[dim,nFr,nSeg] = size(output);
dimS = dim/3;
precision = class(gather(output(1)));
D = genDeltaTransform(nFr, 2, 1, precision);
A = D*D;
D = full(D);
A = full(A);

fgrad = GetFutureGrad(future_layers, curr_layer);

if nSeg==1
    grad = fgrad(1:dimS,:) + fgrad(dimS+1:dimS*2,:)*D + fgrad(dimS*2+1:end,:)*A;
else
    [mask, variableLength] = getValidFrameMask(curr_layer);
    if variableLength
        fgrad = PadShortTrajectory(fgrad, mask, 0);
    end
    
    if variableLength
        future_grad2 = ExtractVariableLengthTrajectory(fgrad, mask);
        for i=1:nSeg
            nFrUtt = size(future_grad2{i},2);
            D = genDeltaTransform(nFrUtt, 2, 1, precision);
            A = D*D;
            D = full(D);
            A = full(A);
            grad(:,1:nFrUtt,i) = future_grad2{i}(1:dimS,1:nFrUtt) + future_grad2{i}(dimS+1:dimS*2,1:nFrUtt)*D + future_grad2{i}(dimS*2+1:end,1:nFrUtt)*A;
        end
        
    else
        fgrad2 = permute(fgrad, [1 3 2]);
        fgrad2S = reshape(fgrad2(1:dimS,:,:), dimS*nSeg,nFr);
        fgrad2D = reshape(fgrad2(dimS+1:dimS*2,:,:), dimS*nSeg,nFr);
        fgrad2A = reshape(fgrad2(dimS*2+1:end,:,:), dimS*nSeg,nFr);
        grad = fgrad2S + fgrad2D*D + fgrad2A*A;
        grad = reshape(grad, dimS, nSeg, nFr);
        grad = permute(grad, [1 3 2]);
    end
end

if 0
    grad2 = zeros(size(grad));
    for i=1:nSeg
        grad2(:,:,i) = fgrad(1:dimS,:,i) + fgrad(dimS+1:dimS*2,:,i)*D + fgrad(dimS*2+1:end,:,i)*A;
    end
end
end