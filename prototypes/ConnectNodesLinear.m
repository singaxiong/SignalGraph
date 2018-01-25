% giving an array of nodes, assume they are connected from the first node
% to the last node. Automatically derive the connections parameters and
% dimensions

function layer = ConnectNodesLinear(layer)

% generate the prev and next properties
for i=1:length(layer)
    if strcmpi(class(layer{i}), 'InputNode')
        % prev represent the input stream index for node input. Don't change it
    else
        layer{i}.prev = -1;
        layer{i}.dim(2) = layer{i-1}.dim(1);    % the input dimension of the current layer is the same as the output dimension of the previous layer. 
    end
    layer{i}.next = 1;
end


end
