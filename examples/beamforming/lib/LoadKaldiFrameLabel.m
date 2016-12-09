function [ali, vocab] = LoadKaldiFrameLabel(label_file)

label_file_mat = [label_file '.mat'];
if exist(label_file_mat)
    load(label_file_mat);
else
     [uttID, alignment, vocab] = readKaldiAlignmentText(label_file);
     usedUttID = {};
     for i=1:length(uttID)
         if isempty(regexp(uttID{i}, 'CH5')); continue; end     % only use channel 5's alignment
         words = ExtractWordsFromString_v2(uttID{i}, '_');
         uttID2{i} = [words{2} '_' words{3}(1:3) '_' words{4}];
         ali.(['U_' uttID2{i}]) = alignment{i};
         usedUttID{end+1} = uttID{i};
     end
     save(label_file_mat, 'ali', 'vocab');
end

end
