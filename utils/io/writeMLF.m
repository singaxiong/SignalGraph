function writeMLF(filename, utt_name, label)

FID = fopen(filename,'w');
fprintf(FID,'#!MLF!#\n');

for j=1:length(utt_name)
    fprintf(FID, '"*/%s.rec"\n', utt_name{j});
    for k=1:length(label{j})
        fprintf(FID, '%s\n', label{j}{k});
    end
    fprintf(FID, '.\n');
end

fclose(FID);