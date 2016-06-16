% This function can read in all the information from the HTK model file
% with specifications of AURORA2 baseline system.

% AUTHOR: XIAO XIONG
% CREATED: 7 Jul, 2006
% LAST MODIFIED: 7 Jul, 2006
function model = readHTKmodel_nosp(file_name,digits_file)

% digits = {'sil','one','two','three','four','five','six','seven','eight','nine','zero','oh'};
if nargin < 2
    digits_file = 'D:\Research\Workspace\Projects\AURORA2\LIB\monophone1';
end
if nargin < 1
    file_name = 'D:\Research\Workspace\Projects\AURORA2\MODEL\HMM08CleanCond_phonePostX\hmm35\models';
end
silence = {'sil','sp','silst'};
FID = fopen(file_name,'r');
digits = my_cat(digits_file);

% Read in the global parameters
fsearch('~o',FID);
tmp = textscan(FID,'%s %d %d',1);
tmp = fgetl(FID); tmp = fgetl(FID);
idx = regexp(tmp,'<'); 
Dimension = str2num(tmp(idx(1)+10:idx(2)-1));        % Dimension of the feature vector
idx = regexp(tmp,'<'); 
feature_type = tmp(idx(3)+1:idx(4)-2);         % type of the feature vector
cov_type = tmp(idx(4)+1:end-1);;             % type of the covariance matrix

frewind(FID);
status = fsearch_line('~v "varFloor1"',FID);
if status == 0
    tmp = fgetl(FID);
    tmp = fgetl(FID);
    tmp = textscan(tmp, '%f', Dimension);
    model.varFloor = tmp{1};
else
    model.varFloor = [];
end

% Read in the digit models
for i=1:length(digits)
    frewind(FID);
    fsearch_line(sprintf('~h "%s"',digits{i}),FID);
    tmp = textscan(FID,'%s',1);
    tmp = textscan(FID,'%s %d',1);
    model.(digits{i}).NUMSTATES = tmp{2}-2;

    for j=1:model.(digits{i}).NUMSTATES
        tmp = textscan(FID,'%s %d',1);
        tmp = textscan(FID,'%s',1);
        if strcmp(tmp{1},'~s')  % this is a silst state
            tmp = textscan(FID,'%s',1);
            model.(digits{i}).NUMMIXES(j) = model.silst.NUMMIXES;
            if model.silst.NUMMIXES == 1
                model.(digits{i}).mean(:,1,j) = model.silst.mean;
                model.(digits{i}).var(:,1,j) = model.silst.var;
                model.(digits{i}).weight(1,j) = model.silst.weight;
            else
                model.(digits{i}).mean(:,:,j) = model.silst.mean;
                model.(digits{i}).var(:,:,j) = model.silst.var;
                model.(digits{i}).weight(:,j) = model.silst.weight;
            end
            continue;
        elseif strcmp(tmp{1},'<MEAN>')
            model.(digits{i}).NUMMIXES(j) = 1;
            fseek(FID,-7,'cof');
        else
            tmp = textscan(FID,'%d',1);
            model.(digits{i}).NUMMIXES(j) = tmp{1};
        end
        for k=1:model.(digits{i}).NUMMIXES(j)
            if model.(digits{i}).NUMMIXES(j)>1
                tmp = textscan(FID,'%s %d',1);
                tmp = textscan(FID,'%f ',1);
                model.(digits{i}).weight(k,j) = tmp{1};
            else
                model.(digits{i}).weight(k,j) = 1;
            end
            tmp = textscan(FID,'%s %d',1);
            tmp = textscan(FID,'%f ',Dimension);
            model.(digits{i}).mean(:,k,j) = tmp{1};

            tmp = textscan(FID,'%s %d',1);
            tmp = textscan(FID,'%f ',Dimension);
            model.(digits{i}).var(:,k,j) = tmp{1};

            tmp = textscan(FID,'%s',1);
            fseek(FID,-length(tmp{1}{1})-1,'cof');
            if strcmp(tmp{1}{1},'<GCONST>')
                tmp = textscan(FID,'%s %f',1);
                model.(digits{i}).GCONST(k,j) = tmp{2};
            end
        end
    end
    % Read in the trainsit matrix
    tmp = textscan(FID,'%s %d',1);
    for j=1:model.(digits{i}).NUMSTATES+2
        tmp = textscan(FID,'%f ',model.(digits{i}).NUMSTATES+2);
        model.(digits{i}).TRANSP(j,:) = tmp{1};
    end
end
fclose(FID);