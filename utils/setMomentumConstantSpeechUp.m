% This function will set momentum to appropriate values such that the
% effective increase in learning rate is constant for each increase of
% momentum. 
% startMomentum is the momentum we want to start with. 
% stopMomentum is the final momentum we want to reach in nSteps. For
% example, if startMomentum=0.5, stopMomentum=0.9, and every step we
% increase momentum by 0.1, then nSteps is equal to 5. 
% Author: Xiong Xiao, NTU, Singapore. 
% 31 May 2015. 
%
function momentum = setMomentumConstantSpeechUp(startMomentum, stopMomentum, nSteps)

% first work out the effective learning rate increase
LR_start = 1/(1-startMomentum);
LR_stop = 1/(1-stopMomentum);

LR_ratio = LR_stop/LR_start;

% compute the effective increase in learning rate in every step
increaseRatio = exp(log(LR_ratio)/(nSteps-1));

% work out the momentum at eaach steps
momentum(1) = startMomentum;
effectiveLR = 1/(1-momentum(1));
for t=2:nSteps-1
    effectiveLR = effectiveLR * increaseRatio;
    momentum(t) = 1-1/effectiveLR;
end
momentum(nSteps)= stopMomentum;