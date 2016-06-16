function layer = KaldiNet2Layer(nnet)

layer{1} = [];
for i=1:length(nnet);
    if strcmpi(nnet{i}.name, 'affinetransform')
        layer{end+1}.name = 'Affine';
        layer{end}.W = nnet{i}.transform;
        layer{end}.b = nnet{i}.bias';
    elseif strcmpi(nnet{i}.name, 'lineartransform')
        layer{end+1}.name = 'LinearTransform';
        layer{end}.W = nnet{i}.transform;
    end
end
