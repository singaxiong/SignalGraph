function csv = readCSV(filename)
FID = fopen(filename);
currLine = fgetl(FID);
idx = find(currLine==',');
field{1} = currLine(1:idx(1)-1);
csv.(field{1}) = {};
for i=2:length(idx)
    field{end+1} = regexprep(currLine(idx(i-1)+1:idx(i)-1), ' ', '_');
    csv.(field{end}) = {};
end
field{end+1} = regexprep(currLine(idx(end)+1:end), ' ', '_');
csv.(field{end}) = {};
nField = length(field);

line_cnt = 0;
while 1
    currLine = fgetl(FID);
    line_cnt = line_cnt + 1;
    PrintProgress(line_cnt, 1e5, 5000);
    
    if currLine == -1
        break;
    end
    
    idx = find(currLine==',');
    for i=1:nField
        if i==1
            idx1 = 1;
        else
            idx1 = idx(i-1)+1;
        end
        if i==nField
            idx2 = length(currLine);
        else
            idx2 = idx(i)-1;
        end
        
        csv.(field{i}){end+1} = currLine(idx1:idx2);
    end
end

fclose(FID);