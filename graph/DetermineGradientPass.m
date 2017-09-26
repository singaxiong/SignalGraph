% For each layer, determine whether we need to pass gradient back
function layer = DetermineGradientPass(layer)

% find the parent of every layer
parent = DetermineLayerParent(layer);

skipBP = zeros(length(layer),1);
for i=1:length(layer)
    if isfield(layer{i}, 'skipBP')
        skipBP(i) = layer{i}.skipBP;
    end
end

update = zeros(length(layer),1);
for i=1:length(layer)
    if isfield(layer{i}, 'update')
        update(i) = layer{i}.update;
    end
end

for i=1:length(layer)
    if IsUpdatableNode(layer{i}.name)==0; continue; end
    if skipBP(i); continue; end
    
    layer{i}.passGradBack = sum(update(parent{i}))>0;
end

end