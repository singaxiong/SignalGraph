%
% compute delta and acceleration features from static features
%
function output = comp_dynamic_feature(feature, delta_order, acc_order)

if nargin<3
    acc_order = 2;
end
if nargin<2
    delta_order = 3;
end

static = feature;
delta = comp_delta(static,delta_order);
acc = comp_delta(delta,delta_order);
output = [static delta acc];
end