%
% compute delta and acceleration features from static features
%
function feature = comp_dynamic_feature2(feature)

D = size(feature,2);
feature(:,D+1:2*D) = comp_delta(feature(:,1:D),3);
feature(:,2*D+1:3*D) = comp_delta(feature(:,D+1:2*D),2);
feature(:,3*D+1:4*D) = comp_delta(feature(:,2*D+1:3*D),2);