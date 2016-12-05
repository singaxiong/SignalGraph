% given a network and some data, compute the mean and variance of the
% network output. This function is usually used to determine the global
% mean and variance normalization parameters in the preprocessing. 
% Xiong Xiao
%
function [W, b] = computeGlobalCMVN(Visible, nUttUsed, para, layer)
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

W = diag(1./std(feat'));
logMel2 = W * feat;
b = -mean(logMel2,2);


end
