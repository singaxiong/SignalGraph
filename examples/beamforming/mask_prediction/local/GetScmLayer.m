function [scm_idx, split] = GetScmLayer(layer)

scm_idx = ReturnLayerIdxByName(layer, 'SpatialCovSplitMask');
if isempty(scm_idx)
    scm_idx = ReturnLayerIdxByName(layer, 'SpatialCovMask');
    split = 0;
else
    split = 1;
end
end