function [y,MSE] = delta2static(x, delta_order, dimS, scale)
[nFr, dim] = size(x);
nDelta = dim/dimS;

if length(delta_order)==1 && nDelta > 1
    delta_order = ones(nDelta, 1) * delta_order;
end

% Initialize the static features
y = x(:,1:dimS);


D = genDeltaTransform(nFr, delta_order(1));
A = D*D;

ScaleD = eye(nFr)*scale(2);
ScaleA = eye(nFr)*scale(3);

R = eye(nFr) + D'*ScaleD*D + A'*ScaleA*A;
p = y + D' * ScaleD * x(:,dimS+1:dimS*2) + A' * ScaleA * x(:,dimS*2+1:dimS*3);
y2 = inv(R) * p;

y = y2;

MSE = [];
if 1
    x2 = comp_dynamic_feature(x(:,1:dimS), 2, 2);
    x3 = comp_dynamic_feature(y, 2, 2);
    MSE(1,1) = sum(sum((x2(:,1:dimS)-x(:,1:dimS)).^2));
    MSE(1,2) = sum(sum((x3(:,1:dimS)-x(:,1:dimS)).^2));
    MSE(2,1) = sum(sum((x2(:,dimS+1:dimS*2)-x(:,dimS+1:dimS*2)).^2));
    MSE(2,2) = sum(sum((x3(:,dimS+1:dimS*2)-x(:,dimS+1:dimS*2)).^2));
    MSE(3,1) = sum(sum((x2(:,dimS*2+1:dimS*3)-x(:,dimS*2+1:dimS*3)).^2));
    MSE(3,2) = sum(sum((x3(:,dimS*2+1:dimS*3)-x(:,dimS*2+1:dimS*3)).^2));
end