function words = ExtractWordsFromString_v2(inputString, space_deliminator)

inWord = 0;
words = {};
if nargin<2
    spaceIdx = regexp(inputString, '\s');
else
    spaceIdx = regexp(inputString, space_deliminator);
end
spaceIndicator = zeros(size(inputString));
spaceIndicator(spaceIdx) = 1;

for k=1:length(inputString)
    if spaceIndicator(k)==1
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
