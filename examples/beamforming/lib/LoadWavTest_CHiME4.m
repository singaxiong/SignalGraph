function wavlist = LoadWavTest_CHiME4(para)

if para.topology.nCh==6
    wavlist = {};
    dataset = {'dt05', 'et05'};
    noise = {'bus', 'caf', 'ped', 'str'};
    type = {'simu', 'real'};
    for i = 1:length(dataset)
        for j=1:length(noise)
            for k=1:length(type)
                wavroot = [para.local.wavroot_noisy '/' dataset{i} '_' noise{j} '_' type{k}];
                tmplist = findFiles(wavroot, 'wav');
                wavlist = [wavlist tmplist];
            end
        end
    end    
else
    wavroot = [para.local.wavroot_noisy '_' num2str(para.topology.nCh) 'ch_track'];
    wavlist = findFiles(wavroot, 'wav');
end

for i=length(wavlist):-1:1
    if ~isempty(regexp(wavlist{i}, 'CH0'))
        wavlist(i) = [];
    end
end

wavlist = reshape(wavlist, para.topology.nCh, length(wavlist)/para.topology.nCh);

% verify that the uttIDs of the same sentence are the same
for i=1:size(wavlist,2)
    for j=1:size(wavlist,1)
        [~,uttID{j}] = fileparts(wavlist{j,i});
        [~,uttID{j}] = fileparts(uttID{j});
    end
    for j=2:size(wavlist,1)
        if strcmpi(uttID{1},uttID{j})==0
            fprintf('Error: uttID is not equal for different channels of the same sentence\n');
            break;
        end
    end
end

end
