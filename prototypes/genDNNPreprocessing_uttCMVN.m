function processing = genDNNPreprocessing_uttCMVN(dim, doCVN)
if nargin<2
    doCVN = 0;
end

if doCVN==0
    processing.name = 'CMN';
else
    processing.name = 'MVN';
end

processing.inputDim = dim;
processing.outputDim = dim;

end