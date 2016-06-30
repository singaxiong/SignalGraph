% set default value for a field if not set yet. 
function para = SetDefaultValue(para, fieldName, defaultValue)
    if ~isfield(para, fieldName)
        para.(fieldName) = defaultValue;
    end
end
