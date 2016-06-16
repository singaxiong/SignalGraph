function weight = read_filter(file_name);
FID = fopen(file_name,'r');
tmp = textscan(FID,'%f ',2);
filter_len = tmp{1}(1);
N_ch = tmp{1}(2);
for i=1:N_ch,
    tmp = textscan(FID,'%f ', filter_len);
    weight(:,i) = tmp{1};
end
fclose(FID);