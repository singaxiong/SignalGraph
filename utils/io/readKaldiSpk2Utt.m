function [spk2utt, nSession] = readKaldiSpk2Utt(filename)

lines = my_cat(filename);

for i=1:length(lines)
    words = ExtractWordsFromString_v2(lines{i});
    fieldname = ['S_' regexprep(words{1},'[.|\-]','_')];
    spk2utt.(fieldname) = {};
    for j=2:length(words)
        spk2utt.(fieldname){end+1} = words{j};
    end
    nSession(i) = length(words)-1;
end
end