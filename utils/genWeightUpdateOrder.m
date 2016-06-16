function weight_update_order = genWeightUpdateOrder(layer, WeightTyingSet)

% define the set of parameters to be updated
weight_update_order = WeightTyingSet;
already_in_set = cell2mat(weight_update_order);

for i=length(layer):-1:1
    if IsUpdatableNode(layer{i}.name)==0; continue; end
    if layer{i}.update == 0; continue; end
    if ismember(i, already_in_set); continue; end
    weight_update_order{end+1} = i;
    already_in_set(end+1) = i;
end

end