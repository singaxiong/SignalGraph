% set default values for function arguments 
    if ~isfield(para, fieldName)
        para.(fieldName) = defaultValue;
    end
end
