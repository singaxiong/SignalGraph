function genModulationTrainTest


para.type = 2;
para.fs = 16000;
para.win_size = 100;
para.win_shift = 50;
para.norm = 'MVN';

flist = findFiles('F:\Dropbox\Projects\ReverbChallenge2014Share\T60Estimation\wav', 'wav');

for i=1:length(flist)
    PrintProgress(i, length(flist), 100);
    wav = wavread(flist{i});
    [modu,moduAvg{i}] = compute_modulation(wav, para);
end

output = sprintf('moduAvg_train_test_MVN.mat');
save(output, 'moduAvg', 'flist');

end
