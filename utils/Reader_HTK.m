function feat = Reader_HTK(files, big_endian, precision)
if nargin<3
    precision = 'single';
end
if nargin<2
    big_endian = 0;
end
for i=1:length(files)
    feat{i} = readHTK(files{i}, [], big_endian,1);
    if strcmpi(precision, 'single')
        feat{i} = single(feat{i});
    end
end
end
