function processing = genDNNPreprocessing_delta(dim, delta_order)
if nargin<2
    delta_order = 2;
end

processing.name = 'delta';
processing.delta_order = delta_order;
processing.inputDim = dim;
processing.outputDim = dim*3;

end