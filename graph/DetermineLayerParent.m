% find the parent of every layer
function parent = DetermineLayerParent(layer)

for i=1:length(layer)
    
    switch lower(layer{i}.name)
        case {'weight2activation', 'input'}
            parent{i} = [];
        otherwise
            immediateParent = i+layer{i}.prev;
            parent{i} = [];
            for j=immediateParent
                parent{i} = [parent{i} j parent{j}];
            end
    end    
    
end

end