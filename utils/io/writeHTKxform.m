function writeHTKxform(fileName, trans, transType)

fileName = dos2unix(fileName);
dim = size(trans.A, 1);
idx = regexp(fileName, '/');
transName = fileName(idx(end)+1:end);

FID = fopen(fileName, 'w');

fprintf(FID, '~a "%s"\n', transName);
fprintf(FID, '<ADAPTKIND>BASE\n');
fprintf(FID, '<BASECLASS>~b "global"\n');
fprintf(FID, '<XFORMSET>\n');
fprintf(FID, '<XFORMKIND>%s\n', transType);
fprintf(FID, '<NUMXFORMS> 1\n');
fprintf(FID, '<LINXFORM> 1\n');
fprintf(FID, '<VECSIZE> %d\n', dim);
fprintf(FID, '<OFFSET>\n');
fprintf(FID, '<BIAS> %d\n', dim);

write_matrix(FID, trans.b');

fprintf(FID, '<BLOCKINFO> 1 %d\n', dim);
fprintf(FID, '<BLOCK> 1\n');
fprintf(FID, '<XFORM> %d %d\n', dim, dim);

write_matrix(FID, trans.A);

fprintf(FID, '<XFORMWGTSET>\n');
fprintf(FID, '<CLASSXFORM> 1 1\n');

fclose(FID);

