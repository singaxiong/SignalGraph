function str = ConcatWordsWithSpace(words, spaceStr)
if nargin<2
    spaceStr = ' ';
end

if length(words)==0
    str = '';
    return;
end

str = words{1};
for i=2:length(words)
    if spaceStr == ' '
        str = [str spaceStr words{i}];
    elseif strcmpi(spaceStr, '\t')
        str = sprintf('%s\t%s', str, words{i});
    else
        fprintf('Unsupported spaceStr: %s\n', spaceStr);
    end
end

end
