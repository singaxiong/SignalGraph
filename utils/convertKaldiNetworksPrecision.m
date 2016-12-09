function kaldiNetwork = convertKaldiNetworksPrecision(kaldiNetwork, precision)

fields = {'transform', 'bias'};

for i=1:length(kaldiNetwork)
    for j=1:length(fields)
        if isfield(kaldiNetwork{i}, fields{j})
            kaldiNetwork{i}.(fields{j}) = cast( kaldiNetwork{i}.(fields{j}), precision );
        end
    end
end
end
