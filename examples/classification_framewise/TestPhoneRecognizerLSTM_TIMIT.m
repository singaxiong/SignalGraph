function TestPhoneRecognizerLSTM_TIMIT()

modeldir = my_dir('nnet');
modelfiles = findFiles(['nnet/' modeldir{1}], 'mat');
modelfiles = sort_nnet_by_itr(modelfiles);      % get the last iteration model

dnn = load(modelfiles{1});
para = dnn.para;
para.IO = RemoveIOStream(para.IO, 2);   % during evaluation, we don't have the label usually. So remove that stream from the configuration
layer = dnn.layer(1:end-2);     % we discard the last two layers that is not useful for ASR

clean_cond = 1;
[Data_cv, ~, para] = LoadData_TIMIT(para, 'test');

para.out_layer_idx = length(layer) + [0];   % you can specify which layers' activation will be outputed


for i=1:length(Data_cv(1).data)
    tmpData(1).data{1} = Data_cv(1).data{i};
    output = FeatureTree2(tmpData, para, layer);
    posterior = output{1}{1};
    
    subplot(2,1,1); imagesc(Data_cv(1).data{i});    title('Feature');
    subplot(2,1,2); imagesc(posterior);       title('Posteriorgram and true label'); hold on
    plot(Data_cv(2).data{i}, 'r'); hold off;
    pause
end

end
