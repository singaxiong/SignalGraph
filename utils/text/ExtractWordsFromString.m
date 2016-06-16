function words = ExtractWordsFromString(inputString)

inWord = 0;
words = {};

for k=1:length(inputString)
    if inputString(k) == ' '
        isSpace = 1;
    elseif ~isempty( regexp(inputString(k), '\s', 'once') )
        isSpace = 1;
    else
        isSpace = 0;
    end
    
    if inWord == 1
        if isSpace == 1
            words{end+1} = inputString(word_start:k-1);
            inWord = 0;
        end
        
    else
        if isSpace == 0
            inWord = 1;
            word_start = k;
        end
    end
end

% if the last symbol is not space and we are still in a word. 
if inWord==1
    words{end+1} = inputString(word_start:end);
end
