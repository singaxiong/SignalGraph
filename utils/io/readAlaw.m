function wav = readAlaw(file_name)

FILE = fopen( file_name );
if FILE < 0
    fprintf('File open error: %s\n', file_name);
    return;
end

data = fread(FILE, 'uint8');
wav = pcma2lin(data);
% plot(wav); pause;
% clear binData;
% binChars = dec2bin(data);
% for i=1:8
%     binData(:,i) = str2num(binChars(:,i));
% end


fclose(FILE);



