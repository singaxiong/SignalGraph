function e = my_splitString(s, pattern)

s2 = regexprep(s, pattern, ' ');
e = ExtractWordsFromString(s2);
