
function grad = B_dynamic_feat(curr_layer, future_layers)
output = curr_layer.a;
[dim,nFr,nSeg] = size(output);
dimS = dim/3;
D = genDeltaTransform(nFr, 2);
A = D*D;
D = full(D);
A = full(A);

fgrad = GetFutureGrad(future_layers, curr_layer);

if nSeg==1
    grad = fgrad(1:dimS,:) + fgrad(dimS+1:dimS*2,:)*D + fgrad(dimS*2+1:end,:)*A;
else
    fgrad2 = permute(fgrad, [1 3 2]);
    fgrad2S = reshape(fgrad2(1:dimS,:,:), dimS*nSeg,nFr);
    fgrad2D = reshape(fgrad2(dimS+1:dimS*2,:,:), dimS*nSeg,nFr);
    fgrad2A = reshape(fgrad2(dimS*2+1:end,:,:), dimS*nSeg,nFr);
    grad = fgrad2S + fgrad2D*D + fgrad2A*A;
    grad = reshape(grad, dimS, nSeg, nFr);
    grad = permute(grad, [1 3 2]);
end

if 0
    grad2 = zeros(size(grad));
    for i=1:nSeg
        grad2(:,:,i) = fgrad(1:dimS,:,i) + fgrad(dimS+1:dimS*2,:,i)*D + fgrad(dimS*2+1:end,:,i)*A;
    end
end
end