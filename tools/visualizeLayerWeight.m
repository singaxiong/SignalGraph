function visualizeLayerWeight(W, blk_m, blk_n, patch_m, orderedByNorm, normalizeWeight)
if nargin<6
    normalizeWeight=0;
end
if nargin<5
    orderedByNorm = 0;
end
if orderedByNorm
    w = sum(W.^2,2);
    [~,idx] = sort(w,'descend');
    W = W(idx,:);
end
if normalizeWeight
    W = MVN(W')';
end


[nh,dim] = size(W);
if nargin<4
    patch_m = 257;
end
patch_n = dim/patch_m;

% colormap(gray);
if nargin<3
    blk_n = 30;
end
if nargin<2
    blk_m = 5;
end
blk_size = blk_m*blk_n;
for i=1:blk_size:nh
    bigImage = [];
    idx_end = min(nh, i+blk_m*blk_n-1);
    border_val = min(min(W(i:idx_end,:)));
    for j=1:blk_m
        subImage = ones(patch_m,1)*border_val;
        for k=1:blk_n
            idx = i+(j-1)*blk_n+k-1;
            if idx>idx_end; 
                currW = zeros(1,dim);
            else
                currW = W(idx,:);
            end
            subImage = [subImage reshape(currW, patch_m, patch_n) ones(patch_m,1)*border_val];
        end
        bigImage = [bigImage;  ones(1, size(subImage,2))*border_val; subImage];
    end
    bigImage = [bigImage;  ones(1, size(subImage,2))*border_val];
%     imagesc(bigImage,[-1 1]*0.6); colorbar;
    imagesc(bigImage); colorbar;
    title(sprintf('Patches %d to %d', i, idx_end));
    pause
end
