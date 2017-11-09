% automatically derive the list of layers that the output of the current layer goes.
function layer = FinishLayer_obj(layer)

for i=1:length(layer); layer{i}.next = []; end
for i=length(layer):-1:1
    if strcmpi(layer{i}.name, 'input'); continue; end
    for j=1:length(layer{i}.prev)
        layer{i+layer{i}.prev(j)}.next(end+1) = -layer{i}.prev(j);
    end
end

end
