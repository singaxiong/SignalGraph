function [tasklist, taskfile] = LoadTaskFile_Reverb(taskFileRoot, dataset, datatype, distance, nCh, roomID)
switch dataset
    case 'train'
        datasetStr = 'tr';
    case 'dev'
        datasetStr = 'dt';
    case 'eval'
        datasetStr = 'et';
    otherwise
        fprintf('Error: unknown dataset: %s!\n', dataset);
        return;
end
if strcmpi(datatype, 'simu')
    datatypeStr = 'Sim';
else
    datatypeStr = 'Real';
end

chStr = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'};

if strcmp(distance, 'cln')
    taskfile = [taskFileRoot '/' num2str(nCh) 'ch/' datatypeStr 'Data_' datasetStr '_for_' ...
        distance '_room' num2str(roomID)];
    tasklist = my_cat(taskfile)';
else
    for i=1:nCh
        if strcmp(dataset, 'train')
            taskfile{i} = [taskFileRoot '/' num2str(nCh) 'ch/' datatypeStr 'Data_' datasetStr '_for_' ...
                num2str(nCh) 'ch_' chStr{i}];
        else
            taskfile{i} = [taskFileRoot '/' num2str(nCh) 'ch/' datatypeStr 'Data_' datasetStr '_for_' ...
                num2str(nCh) 'ch_' distance '_room' num2str(roomID) '_' chStr{i}];
        end
        tasklist(i,:) = my_cat(taskfile{i});
    end
end
end
