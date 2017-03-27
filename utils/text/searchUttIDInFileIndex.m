function content = searchUttIDInFileIndex(index, key)

key2 = ['U_' regexprep(key, '-', '_')];
if isfield(index, key2)
    content = index.(key2);
else
    content = [];
end

end
