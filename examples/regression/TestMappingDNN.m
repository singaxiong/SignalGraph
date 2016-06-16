function TestMappingDNN()

modeldir = my_dir('nnet');
modelfiles = findFiles(['nnet/' modeldir{1}], 'mat');
modelfiles = sort_nnet_by_itr(modelfiles);      % get the last iteration model

dnn = load(modelfiles{1});
para = dnn.para;
layer = dnn.layer;

[~, Data_cv, para] = LoadData_AU4_Mapping(para);

para.out_layer_idx = length(layer) + [-2 -3];   % you can specify which layers' activation will be outputed
output = FeatureTree2(Data_cv, para, layer);

for i=1:length(output)
    enhanced{i} = output{i}{1};
end

for i=1:length(enhanced)
    subplot(3,1,1); imagesc(Data_cv(1).data{i});    title('Noisy MFCC');
    subplot(3,1,2); imagesc(enhanced{i});          title('Enhanced MFCC');
    subplot(3,1,3); imagesc(Data_cv(2).data{i});       title('Clean MFCC');
    pause
end    

end
