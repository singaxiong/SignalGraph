function writeKaldiFeature(outputfile, uttID, fbank, nUttPerArchive, compress)
if nargin<5
    compress = 0;
end

nArk = 1;
for i=1:nUttPerArchive:length(uttID)
    idx2 = min(i+nUttPerArchive-1, length(uttID));
    filename = [outputfile '.' num2str(nArk) '.ark'];
    writeKaldiArchiveText(filename, uttID(i:idx2), fbank(i:idx2));
    
    if compress
        gzip(filename);
        delete(filename);
    end
    nArk = nArk + 1;
end

