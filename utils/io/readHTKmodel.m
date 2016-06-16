% This function can read in all the information from the HTK model file
% with specifications of AURORA2 baseline system.

% AUTHOR: XIAO XIONG
% CREATED: 7 Jul, 2006
% LAST MODIFIED: 7 Jul, 2006
function model = readHTKmodel(file_name,digits_file)

% digits = {'sil','one','two','three','four','five','six','seven','eight','nine','zero','oh'};
if nargin < 2
    digits_file = 'E:\Research\Workspace\Projects\Normalization\AURORA2\LIB\words4';
end
if nargin < 1
    file_name = 'E:\Research\Workspace\Projects\Normalization\AURORA2\MODEL\MFCC_0DA\hmm20\models';
end
silence = {'sil','sp','silst'};
FID = fopen(file_name,'r');
digits = my_cat(digits_file);
digits2 = {};
for i=1:length(digits)
    if length(digits{i})>0
        digits2{end+1} = digits{i};
    end
end
digits = digits2;

% Read in the global parameters
fsearch('~o',FID);
tmp = textscan(FID,'%s %d %d',1);
tmp = fgetl(FID); tmp = fgetl(FID);
idx = regexp(tmp,'<'); 
Dimension = str2num(tmp(idx(1)+10:idx(2)-1));        % Dimension of the feature vector
idx = regexp(tmp,'<'); 
feature_type = tmp(idx(3)+1:idx(4)-2);         % type of the feature vector
cov_type = tmp(idx(4)+1:end-1);;             % type of the covariance matrix


% Read in the silst model
endOfFile = fsearch('~s "silst"',FID);
if endOfFile==0
    model.silst.NUMSTATES = 1;
    tmp = textscan(FID,'%s %d',1);
    if strcmp(tmp{1},'<MEAN>')
        model.silst.NUMMIXES = 1;
        fseek(FID,-7,'cof');
    else
        model.silst.NUMMIXES = tmp{2};
    end
    % model.('silst').NUMMIXES = tmp{2};
    for i=1:model.silst.NUMMIXES
        if model.silst.NUMMIXES>1
            tmp = textscan(FID,'%s %d',1);
            tmp = textscan(FID,'%f ',1);
            model.silst.weight(i) = tmp{1};
        else
            model.silst.weight(i) = 1;
        end
        
        tmp = textscan(FID,'%s %d',1);
        tmp = textscan(FID,'%f ',Dimension);
        model.silst.mean(:,i) = tmp{1};
        
        tmp = textscan(FID,'%s %d',1);
        tmp = textscan(FID,'%f ',Dimension);
        model.silst.var(:,i) = tmp{1};
        
        tmp = textscan(FID,'%s %f',1);
        model.silst.GCONST(:,i) = tmp{2};
    end
    model.silst.weight = model.silst.weight(:);
end

% Read in the sp model
frewind(FID);
endOfFile = fsearch(sprintf('~h "sp"'),FID);
if endOfFile==0
    fsearch(sprintf('<TRANSP> 3'),FID);
    model.sp = model.silst;
    model.sp.NUMSTATES = 1;
    for j=1:model.sp.NUMSTATES+2
        tmp = textscan(FID,'%f ',3);
        model.sp.TRANSP(j,:) = tmp{1};
    end
end

% Read in the digit models
for i=1:length(digits)
    frewind(FID);
    fsearch(sprintf('~h "%s"',digits{i}),FID);
    tmp = textscan(FID,'%s',1);
    tmp = textscan(FID,'%s %d',1);
    
    if strcmp(digits{i},'@')
        currDigit = 'AA';
    else
        currDigit = digits{i};
    end
    model.(currDigit).NUMSTATES = tmp{2}-2;

    for j=1:model.(currDigit).NUMSTATES
        tmp = textscan(FID,'%s %d',1);
        tmp = textscan(FID,'%s',1);
        if strcmp(tmp{1},'~s')  % this is a silst state
            tmp = textscan(FID,'%s',1);
            model.(currDigit).NUMMIXES(j) = model.silst.NUMMIXES;
            if model.silst.NUMMIXES == 1
                model.(currDigit).mean(:,1,j) = model.silst.mean;
                model.(currDigit).var(:,1,j) = model.silst.var;
                model.(currDigit).weight(1,j) = model.silst.weight;
            else
                model.(currDigit).mean(:,:,j) = model.silst.mean;
                model.(currDigit).var(:,:,j) = model.silst.var;
                model.(currDigit).weight(:,j) = model.silst.weight;
            end
            continue;
        elseif strcmp(tmp{1},'<MEAN>')
            model.(currDigit).NUMMIXES(j) = 1;
            fseek(FID,-7,'cof');
        else
            tmp = textscan(FID,'%d',1);
            model.(currDigit).NUMMIXES(j) = tmp{1};
        end
        for k=1:model.(currDigit).NUMMIXES(j)
            if model.(currDigit).NUMMIXES(j)>1
                tmp = textscan(FID,'%s %d',1);
                tmp = textscan(FID,'%f ',1);
                model.(currDigit).weight(k,j) = tmp{1};
            else
                model.(currDigit).weight(k,j) = 1;
            end
            tmp = textscan(FID,'%s %d',1);
            tmp = textscan(FID,'%f ',Dimension);
            model.(currDigit).mean(:,k,j) = tmp{1};

            tmp = textscan(FID,'%s %d',1);
            tmp = textscan(FID,'%f ',Dimension);
            model.(currDigit).var(:,k,j) = tmp{1};

            tmp = textscan(FID,'%s %f',1);
            model.(currDigit).GCONST(k,j) = tmp{2};
        end
    end
    % Read in the trainsit matrix
    tmp = textscan(FID,'%s %d',1);
    for j=1:model.(currDigit).NUMSTATES+2
        tmp = textscan(FID,'%f ',model.(currDigit).NUMSTATES+2);
        model.(currDigit).TRANSP(j,:) = tmp{1};
    end
end
fclose(FID);