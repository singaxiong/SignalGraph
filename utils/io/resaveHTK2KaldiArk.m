function resaveHTK2KaldiArk(flist, uttID, outdir, nUttPerArk, archive_name_prefix)
my_mkdir(outdir);
nUtt = length(flist);
nArk = ceil(nUtt/nUttPerArk);

for ai = 1:nArk
    PrintProgress(ai, nArk, 1, 'Resave features in Kaldi archive format');
    
    startIdx = (ai-1)*nUttPerArk + 1;
    stopIdx = min(nUtt, ai*nUttPerArk);
    ark_name = [outdir '/' archive_name_prefix '.' num2str(ai)];
    saveHTK2KaldiArk(flist(startIdx:stopIdx), uttID(startIdx:stopIdx), ark_name);
end
end


function saveHTK2KaldiArk(flist, uttID, ark_name)
for i=1:length(flist)
    spec{i} = readHTK(flist{i});
end
writeKaldiArchiveText(ark_name, uttID, spec);
gzip(ark_name);        delete(ark_name);
end
