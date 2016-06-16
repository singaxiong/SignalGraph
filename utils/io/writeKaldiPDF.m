function writeKaldiPDF(file_name, pdfID, uttID)
if length(pdfID) ~= length(uttID)
    fprintf('Error: number of pdfID %d is not the same as the number of utterances %d!\n', length(pdfID), length(uttID));
    return;
end

FID = fopen(file_name, 'w');

for i=1:length(pdfID)
    PrintProgress(i, length(pdfID), 1000);
    fprintf(FID, '%s ', uttID{i});
    write_matrix(FID, pdfID{i});
end

fclose(FID);