function [y,LL] = delta2static_ML(mu, variance, delta_order, dimS)
[nFr, dim] = size(mu);
nDelta = dim/dimS;

if length(delta_order)==1 && nDelta > 1
    delta_order = ones(nDelta, 1) * delta_order;
end

D = genDeltaTransform(nFr, delta_order(1));
A = D*D;

for i = 1:dimS
    ScaleS = diag(1./variance(:,i));
    ScaleD = diag(1./variance(:,i+dimS));
    ScaleA = diag(1./variance(:,i+dimS*2));
    
    R = ScaleS + D'*ScaleD*D + A'*ScaleA*A;
    p = ScaleS * mu(:,i) + D' * ScaleD * mu(:,dimS+i) + A' * ScaleA * mu(:,dimS*2+i);
    y(:,i) = inv(R) * p;
end

if 1
    X = comp_dynamic_feature(mu(:,1:dimS), 2, 2);
    Y = comp_dynamic_feature(y, 2, 2);
    XE = (X-mu).^2 ./ variance;
    YE = (Y-mu).^2 ./ variance;
    
    LL(:,1) = mean(XE);
    LL(:,2) = mean(YE);
    
end
end
