function layer = InitDCTLayer(layer, para)

log_idx = ReturnLayerIdxByName(layer, 'log');
for i=1:2
    if strcmpi(layer{log_idx+i}.name, 'affine')
        dct_idx = log_idx + i;
        break;
    end
end

nFbank = layer{dct_idx}.dim(2);
nCep = layer{dct_idx}.dim(1);

layer{dct_idx}.W = GetDctMatrix(nFbank, nCep);
layer{dct_idx}.b = zeros(nCep,1);
end
