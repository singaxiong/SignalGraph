% This function can read in all the information from the HTK model file
% with specifications of AURORA2 baseline system.

% AUTHOR: XIAO XIONG
% CREATED: 7 Jul, 2006
% LAST MODIFIED: 7 Jul, 2006
function model = readGMMs(file_name)

FID = fopen(file_name, 'r');
if FID < 1
    fprintf('Failed to open the model file\n');
    return;
end

tmp = textscan(FID,'%s%d\n',1);
Dim = tmp{2};
fsearch('~v "varFloor1"',FID); fgetl(FID);fgetl(FID);
tmp = textscan(FID,'%n',Dim);
model.varFloor = tmp{1};
fgetl(FID);

modelCnt = 1;
while(fsearch('~g',FID)==0)
    tmp = textscan(FID,'%s',1);
    model.gmm{modelCnt}.name = regexprep(tmp{1}{1},'"','');
    fgetl(FID);fgetl(FID);
    [model.gmm{modelCnt}.prior, model.gmm{modelCnt}.mean, model.gmm{modelCnt}.var] = read_GMM(FID,1);
    fgetl(FID);
    modelCnt = modelCnt+1;
end

fclose(FID);

