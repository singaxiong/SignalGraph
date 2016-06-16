function rttm = LoadRTTM(filename)

matfile = [filename '.mat'];
if exist(matfile, 'file')
    load(matfile);
    return;
end

fprintf('Reading RTTM from file\n');
rttm = readRTTM(filename);

fprintf('Converting strings into numbers\n');
rttm.dur = str2double(rttm.dur);
rttm.confidence = str2double(rttm.confidence);
rttm.tbeg = str2double(rttm.tbeg);
rttm.channel = str2double(rttm.channel);

save(matfile, 'rttm');

end