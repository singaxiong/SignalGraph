% giving an array of nodes, assume they are connected from the first node
% to the last node. Automatically derive the connections parameters and
% dimensions

function layer = ConnectGraph(layer)

% generate connections to the immediate parent nodes (prev)
for i=1:length(layer)
    if strcmpi(class(layer{i}), 'InputNode') || strcmpi(class(layer{i}), 'Weight2ActivationNode')
        % no prev for these nodes
    else
        if isempty(layer{i}.prev)
            layer{i}.prev = -1;
        end
    end
end

% generate connections to immediate children nodes (next)
for i=1:length(layer); layer{i}.next = []; end
for i=length(layer):-1:1
    if strcmpi(class(layer{i}), 'InputNode') || strcmpi(class(layer{i}), 'Weight2ActivationNode'); continue; end
    for j=1:length(layer{i}.prev)
        layer{i+layer{i}.prev(j)}.next(end+1) = -layer{i}.prev(j);
    end
end

% automatically infer the dimensions of the current layer
for i=1:length(layer)
    if strcmpi(class(layer{i}), 'InputNode') || strcmpi(class(layer{i}), 'Weight2ActivationNode')
        continue;
    end
    prev = layer{i}.prev;
    absPrev = prev + i;
    
    for j=1:length(prev)
        prevOutDims(:,j) = layer{absPrev(j)}.dim(1:2);
    end
    layer{i}.dim(3:4) = prevOutDims(:,1)';    % the input dimension of the current layer is the same as the output dimension of the previous layer.
    % currently use the first prev layer's dims. Will implement smarter way
    % to choose the most appropriate previous layer's dims.  
    
end

end
