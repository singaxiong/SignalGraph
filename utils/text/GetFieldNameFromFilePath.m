function fieldName = GetFieldNameFromFilePath(filepath, nStage)
if nargin<2
    nStage = 1;
end

orig_filepath = filepath;
[filepath, uttID] = fileparts(filepath);
fieldName = uttID;
for i=2:nStage
    [filepath, uttID] = fileparts(filepath);
    fieldName = [uttID '_' fieldName];
    if isempty(filepath)
        break;
    end
end
fieldName = ['U_' fieldName];
fieldName = regexprep(fieldName, '-', '_');


end
