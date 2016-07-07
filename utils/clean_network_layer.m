function layer = clean_network_layer(layer)
fields = {'a', 'grad', 'grad_W', 'grad_b', 'acc', 'weights', 'grad_W_raw', 'grad2', 'idx', 'X2', 'ft', 'it', 'ot', 'Ct_raw', 'Ct', 'Ct0', 'ht0', 'post'};
for i=1:length(layer)
    for j=1:length(fields)
        if isfield(layer{i}, fields{j})
            layer{i} = rmfield(layer{i}, fields{j});
        end
    end
end
end