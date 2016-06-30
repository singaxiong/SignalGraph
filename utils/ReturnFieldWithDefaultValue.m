% set default value for a field if not set yet. 
function value = ReturnFieldWithDefaultValue(para, fieldName, defaultValue)
    if ~isfield(para, fieldName)
        value = defaultValue;
    else
        value = para.(fieldName);
    end
end

