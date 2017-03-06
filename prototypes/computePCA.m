
function [W, b] = computePCA(Visible, nUttUsed, para, layer)
if exist('nUttUsed')==0 || length(nUttUsed)==0
    nUttUsed = 500;
end
nUtt = length(Visible(1).data);
if nUtt>nUttUsed
    step = ceil(nUtt/nUttUsed);
    for i=1:length(Visible)
        Visible(i).data = Visible(i).data(1:step:end);
    end
end

para.out_layer_idx = length(layer);
para.output = 'dummy';
para = ParseOptions2(para);
output = FeatureTree2(Visible, para, layer);

if para.NET.variableLengthMinibatch
    for i=1:length(output)
        featTmp = gather(output{i}{1});
        [featTmp2, mask, variableLength] = ExtractVariableLengthTrajectory(featTmp);
        feat{i} = cell2mat(featTmp2);
    end
else
    for i=1:length(output)
        feat{i} = gather(output{i}{1});
        [D,T,N] = size(feat{i});
        if N>1
            feat{i} = reshape(feat{i},D,T*N);
        end
    end
end
feat = cell2mat(feat);

% [coeff, scores, latent] = princomp(feat','econ');
% tmp=cumsum(latent)./sum(latent);
% coeff = princomp(feat','econ');
% W = coeff(:,1:para.topology.pcaDim)';
% b = -W*mean(feat,2);
% 
% [coeff1, scores1, latent1] = pca(feat');
% cov1 = cov(feat');
% [V1,D1] = eig(cov1);
% D2 = diag(D1);

fprintf('Load %d utts feats, begin pca ...', nUttUsed);
[coeff, ~, latent] = pca(feat');
tmp=cumsum(latent)./sum(latent);
idx = find(tmp == 0.95);
fprintf('End of PCA, select %d can cover 99.9 percent', idx);
% W = coeff(:,1:para.topology.pcaDim)';
W = coeff(:,1:1000)';
b = -W*mean(feat,2);

save(['PCA_U' num2str(nUttUsed) '_W_B.mat'], 'W', 'b', 'latent');

end
