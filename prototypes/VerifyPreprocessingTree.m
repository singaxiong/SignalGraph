% verify that the global MVN is correct. 
% Xiong Xiao
function [processing] = VerifyPreprocessingTree(layer, Visible, para, nUttUsed)
if exist('nUttUsed')==0 || isempty(nUttUsed)
    nUttUsed = 500;
end

[W, b] = computeGlobalCMVN(Visible, nUttUsed, para, layer);

plot(-b); hold on;
plot(1./diag(W)); hold off
end

