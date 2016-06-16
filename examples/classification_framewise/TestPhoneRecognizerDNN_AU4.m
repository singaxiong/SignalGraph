function TestPhoneRecognizerDNN()

modeldir = my_dir('nnet');
modelfiles = findFiles(['nnet/' modeldir{1}], 'mat');
modelfiles = sort_nnet_by_itr(modelfiles);      % get the last iteration model

dnn = load(modelfiles{1});
para = dnn.para;
layer = dnn.layer(1:end-2);     % we discard the last two layers that is not useful for ASR

clean_cond = 1;
[~, Data_cv, para] = LoadData_AU4(para, clean_cond);

para.out_layer_idx = length(layer) + [0];   % you can specify which layers' activation will be outputed
output = FeatureTree2(Data_cv, para, layer);

for i=1:length(output)
    posteriogram{i} = output{i}{1};
end

for i=1:length(posteriogram)
    subplot(2,1,1); imagesc(Data_cv(1).data{i});    title('MFCC');
    subplot(2,1,2); imagesc(posteriogram{i});       title('Posteriorgram and true label'); hold on
    plot(Data_cv(2).data{i}, 'r');
    pause
end    

end
