% feat is a cell array of matrix, each with TxD size. T is the number of
% frames and D is the number of dimensions. 

function writeKaldiArchiveText(file_name, uttID, feat)

FID = fopen(file_name, 'W');

for i=1:length(uttID)
    PrintProgress(i, length(uttID), 500, 'Writing Kaldi archive');
    fprintf(FID, '%s [ ', uttID{i});
    write_matrix(FID, feat{i}, 1);
    fprintf(FID, ' ]\n');
end

fclose(FID);