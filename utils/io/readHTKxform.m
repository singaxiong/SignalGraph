function tran = readHTKxform(file_name)

fprintf('Now reading transform from %s\n', file_name);

FID = fopen(file_name,'r');

if fsearch2('<NUMXFORMS>', FID)==0
    nXform = str2num(fgetl(FID));
else
    fprintf('Error: cannot find NUMXFORMS\n');
end

if nXform ==0
    tran.A = [];
    tran.b = [];
    return;
end

for i=1:nXform
    line = fgetl(FID);
    line = fgetl(FID);
    tmp = textscan(line, '%s %d');
    dim = tmp{2};

    line = fgetl(FID);
    line = fgetl(FID);
    line = fgetl(FID);
    tmp = textscan(line, '%f', dim);
    tran.b(:,i) = tmp{1};

    while 1
        line = fgetl(FID);
        if length(regexp(line, '<XFORM>'))
            break;
        end
    end
    tran.A(:,:,i) = read_matrix(FID, dim, dim);
end

fclose(FID);