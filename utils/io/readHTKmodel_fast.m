% This function can read in all the information from the HTK model file
% with specifications of AURORA2 baseline system.

% AUTHOR: XIAO XIONG
% CREATED: 7 Jul, 2006
% LAST MODIFIED: 7 Jul, 2013
function model = readHTKmodel_fast(file_name, skip_hmm)
if nargin<2
    skip_hmm = 0;
end
silence = {'sil','sp','silst'};

FID = fopen(file_name,'r');


line = fgetl(FID);
while 1
%     fprintf('%s\n', line);
%     if regexp(line, 'silst')>0
%         pause(.1);
%     end
    
    if line(1) == '~'
        switch line(2)
            case 'o'
                % Read in meta information
                while 1
                    line = fgetl(FID);
                    words = ExtractWordsFromString_v2(line);
                    if strcmp(words{1}, '<STREAMINFO>')
                        model.nStream = str2num(words{2});
                        model.dim = str2num(words{3});
                    elseif strcmp(words{1}, '<VECSIZE>')
                        % Do nothing
                    elseif line(1) == '~'
                        break;
                    end
                end
                
            case 'v'
                % Read in the variance floor
                modelName = line(5:end-1);
                line = fgetl(FID);
                line = fgetl(FID);
                model.(modelName).var = str2num(line);
                line = fgetl(FID);
                
            case 's'
                % Read in the state info
                modelName = line(5:end-1);
                model.(modelName).nMix = 1;
                [model.(modelName) line] = ReadSingleState(FID, model.(modelName));
                
            case 'h'
                if skip_hmm == 1
                    return;
                end
                
                % Read in an HMM model
                modelName = line(5:end-1);
                modelName = regexprep(modelName, '-|+', '_');
                line = fgetl(FID);
                line = fgetl(FID);
                words = ExtractWordsFromString_v2(line);
                model.(modelName).nState = str2num(words{2})-2;
                
                % Read in the states emission probabilities
                line = fgetl(FID);
                for i=1:model.(modelName).nState
                    stateName = sprintf('state%d',i);
                    model.(modelName).(stateName).nMix = 1;
                    [model.(modelName).(stateName) line] = ReadSingleState(FID, model.(modelName).(stateName));
                end
                
                % Read in the transition matrix
                if strcmp(line(1:2), '~t')
                    model.(modelName).transition = line(5:end-1);
                else
                    model.(modelName).transition = read_matrix(FID, model.(modelName).nState+2, model.(modelName).nState+2);
                end
                line = fgetl(FID);
                line = fgetl(FID);
                
            case 't'
                % Read in a transition matrix
                modelName = line(5:end-1);
                line = fgetl(FID);
                words = ExtractWordsFromString_v2(line);
                nState = str2num(words{2});
                model.transition.(modelName) = read_matrix(FID, nState, nState);
                line = fgetl(FID);
                
            otherwise
                fprintf('Error: Unknown object type: %s\n', line(2));
                break;
        end
    else
        break;
    end
end
fclose(FID);
end

%% 
function [state line] = ReadSingleState(FID, state)

line = fgetl(FID);
if strcmp(line, '~s "silst"')
    state.isSILST = 1;
    line = fgetl(FID);
    return;
else
    words = ExtractWordsFromString_v2(line);
    if strcmp(words{1}, '~s')
        state.name = words{2};
        line = fgetl(FID);
        return;
    end
end
state.isSILST = 0;
words = ExtractWordsFromString_v2(line);
if strcmp(words{1}, '<NUMMIXES>')
    state.nMix = str2num(words{2});
    line = fgetl(FID);
else
    state.nMix = 1;
end

for i=1:state.nMix
    if state.nMix>1
        % Read mixture weight
        idx = regexp(line, ' ');
        state.prior(i) = str2num(line(idx(2):end));
%         words = ExtractWordsFromString(line);
%         state.prior(i) = str2num(words{3});
    else
        state.prior(i) = 1;
    end
    
    while 1
        if length(regexp(line, '<MEAN>'))>0
            break;
        end
        line = fgetl(FID);
    end
    line = fgetl(FID);
    state.mean(:,i) = str2num(line);
    
    line = fgetl(FID);
    line = fgetl(FID);
    state.var(:,i) = str2num(line);
    
    line = fgetl(FID);
    idx = regexp(line, ' ');
    if strcmp(line(1:idx(1)-1), '<GCONST>')
        state.gonst(i) = str2num(line(idx(1)+1:end));
%     words = ExtractWordsFromString(line);
%     if strcmp(words{1}, '<GCONST>')
%         state.gonst(i) = str2num(words{2});
        line = fgetl(FID);
    end
end

end
