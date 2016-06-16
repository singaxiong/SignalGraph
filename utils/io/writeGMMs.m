% This function can read in all the information from the HTK model file
% with specifications of AURORA2 baseline system.

% AUTHOR: XIAO XIONG
% CREATED: 7 Jul, 2006
% LAST MODIFIED: 7 Jul, 2006
function writeGMMs(file_name, model_name, weights, means, vars, varFloor)
FID = fopen(file_name,'w');
if FID<1
    return;
end

stateAsClass = 0;
dim = size(means{1},1);
fprintf(FID,'<VECSIZE> %d\n', dim);
fprintf(FID,'~v varFloor1\n');
fprintf(FID,'<Variance> %d\n', dim);
write_matrix(FID, varFloor);
N_GMM = length(means);
for i=1:N_GMM
    if stateAsClass
        tmp = mod(i,3);
        if tmp == 0
            tmp = 3;
        end
        fprintf(FID,'~g "%s%d"\n', model_name{i},tmp);
    else
        fprintf(FID,'~g "%s"\n', model_name{i});
    end
    
    fprintf(FID,'<BEGINGMM>\n');
    N_Mix = size(means{i},2);
    write_GMM(FID,weights(1:N_Mix,i),means{i},vars{i},1);
    fprintf(FID,'<ENDGMM>\n');
end
fclose(FID);
