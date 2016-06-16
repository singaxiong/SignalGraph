function preparedata

data = load('F:\Dropbox\Workspace\Projects2\Packages\DNN_Demo_2015_08_08\train_clean_multi_tiny.mat');

for i=1:length(data.noisy_list)
    [~,uttID] = fileparts(data.noisy_list{i});
    utt_list{i} = uttID;
end

for i=1:length(data.mfcc_clean_train)
    mfcc_clean_train{i} = single(data.mfcc_clean_train{i}(:,1:13)');
    mfcc_noisy_train{i} = single(data.mfcc_noisy_train{i}(:,1:13)');
end
for i=1:length(data.mfcc_clean_test)
    mfcc_clean_test{i} = single(data.mfcc_clean_test{i}(:,1:13)');
    mfcc_noisy_test{i} = single(data.mfcc_noisy_test{i}(:,1:13)');
end

[label_train phones] = alignment2phonePosterior(data.alignment_train, [], 3);
[label_test] = alignment2phonePosterior(data.alignment_test, [], 3);

save('mfcc_label.mat', 'utt_list', 'mfcc_clean_train', 'mfcc_noisy_train', 'mfcc_clean_test', 'mfcc_noisy_test', 'label_train', 'label_test', 'phones');

end