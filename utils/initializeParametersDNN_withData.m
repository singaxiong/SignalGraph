function init_layer = initializeParametersDNN_withData(visible, target, para)

theta = initializeParametersDNN(para.layerSize, para);    % Use random initialization
init_layer = DNNweights_vec2layer(theta, para);

if isfield(para, 'sparseInput') && para.sparseInput == 1
    % adjust the initial weights
    para.preprocessing{end+1}.name = 'fulltify';
    para.preprocessing{end}.inputDim = para.preprocessing{end-1}.outputDim;
    para.preprocessing{end}.outputDim = para.preprocessing{end}.inputDim;
end

for i=2:length(init_layer)-1
    processing = DNNweights2featurePiple_v2(init_layer(1:i), para);
    processing = [para.preprocessing processing(1:end-1)];
    
    before_sigmoid = FeaturePipe(visible, processing);
    std_z = std(mat2vec(before_sigmoid));
    
    init_layer{i}.W = init_layer{i}.W / std_z;  % we normalize the first hidden layer's input to have unit variance.
end

if strcmp(para.NodeType{end}, 'softmax')
    processing = DNNweights2featurePiple_v2(init_layer, para);
    processing = [para.preprocessing processing(1:end-1)];
    
    before_softmax = FeaturePipe(visible, processing);
    std_z_last = std(mat2vec(before_softmax));
    
    init_layer{end}.W = init_layer{end}.W /std_z_last * 5;  % normalize the variance before the softmax to have variance = 25.
end

if strcmp(para.NodeType{end}, 'linear')
    processing = DNNweights2featurePiple_v2(init_layer, para);
    processing = [para.preprocessing processing(1:end-2)];
    
    last_hidden_out = FeaturePipe(visible, processing);
    processed_target = FeaturePipe(target, para.target_processing);
    paraLinear.notest = 1;
    paraLinear.L2weight = 1e-3;
    [linear_processing, cost] = train_linearMapping( last_hidden_out, processed_target, [], [], paraLinear);
    init_layer{end}.W = linear_processing{1}.transform;
    init_layer{end}.b = linear_processing{1}.bias;
    
    if 0
        processing = DNNweights2featurePiple_v2(init_layer, para);
        processing = [para.preprocessing processing];
        predicted = FeaturePipe(visible, processing);
        imagesc([ processed_target(:,1:1000); predicted(:,1:1000)]);
    end
end
end