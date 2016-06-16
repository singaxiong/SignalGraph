function write_filter(weight, file_name);
[filter_len, N_ch] = size(weight);
FID = fopen(file_name,'w');

fprintf(FID, '%f\n',filter_len);
fprintf(FID, '%f\n',N_ch);

for i=1:N_ch, 
    for j=1:filter_len
        fprintf(FID, '%f ',weight(j,i));
    end
    fprintf(FID,'\n');
end
fclose(FID);