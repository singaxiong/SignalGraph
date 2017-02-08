function wavlist = Tasklist2wavlist(wavroot, tasklist, dataset, datatype)
if strcmpi(datatype, 'simu')
    datatypeStr = 'REVERB_WSJCAM0';
    switch dataset
        case 'train'
            datasetStr = 'tr/data';
        case 'dev'
            datasetStr = 'dt/data';
        case 'eval'
            datasetStr = 'et/data';
        otherwise
            fprintf('Error: unknown dataset: %s!\n', dataset);
            return;
    end
else
    datatypeStr = 'MC_WSJ_AV';
    datasetStr = dataset;
    datasetStr(1) = upper(datasetStr(1));
end

wavdir = [wavroot '/' datatypeStr '_' datasetStr];
for i=1:size(tasklist,1)
    for j=1:size(tasklist,2)
    wavlist{i,j} = [wavdir '/' tasklist{i,j}];
end
end