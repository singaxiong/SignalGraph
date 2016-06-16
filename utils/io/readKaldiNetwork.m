function modules = readKaldiNetwork(file_name)
% file_name = 'tri1a_dnn_ctx5_plain_1500b/nnet/nnet_429_1500_1500_1500_1500_2584_iter14_learnrate3.125e-05_tr68.77_cv54.38';
FID = fopen(file_name, 'r');

modules = {};
line = fgetl(FID);
while 1
    if line(1) == '<'
        if length(regexp(line, '<Nnet>'))>0 || length(regexp(line, '</Nnet>'))>0
            line = fgetl(FID);
            continue;
        end
        
        fprintf('Reading %s - %s\n', line, datestr(now));
        idx = regexp(line, '>');
        modules{end+1}.name = line(2:idx(1)-1);
        tmp = textscan(line(idx(1)+1:end), '%d %d');
        modules{end}.inputDim = tmp{2};
        modules{end}.outputDim = tmp{1};
        
        switch modules{end}.name
            case {'affinetransform', 'AffineTransform'}
                modules{end}.transform = readKaldiTransform(FID, modules{end}.outputDim);
                modules{end}.bias = readKaldiTransform(FID, 1);
                
            case {'lineartransform', 'LinearTransform'}
                modules{end}.transform = readKaldiTransform(FID, modules{end}.outputDim);
                
            case {'sigmoid', 'softmax', 'Sigmoid', 'Softmax'}
                % Do nothing
                
            case {'Splice', 'splice', 'AddShift', 'addshift', 'rescale', 'Rescale'}
                modules{end}.transform = readKaldiTransform(FID, 1);

            otherwise
                fprintf('Error: unknown processing step: %s~\n', modules{end}.name);
                break;
        end
    else
        break;
    end
    line = fgetl(FID);
end
fclose(FID);


%%
function transform = readKaldiTransform(FID, M)
line = fgetl(FID);

% if length(regexp(line, '\]'))>0
%     line = fgetl(FID);
%     line = regexprep(line, '\[', '');
% else
idx = regexp(line, '\[');
if length(idx) > 0
    line = line(idx+1:end);
    if length(regexprep(line, ' ', ''))==0
        line = fgetl(FID);
    end
else
    line = fgetl(FID);
end
for i=1:M-1
    transform(i,:) = str2num(line);
    line = fgetl(FID);
end
if length(regexprep(line, '\]| ', '')) > 0
    if length(regexp(line, '\]'))==0   
        line2 = fgetl(FID); % Read the ]
    end
    line = regexprep(line, '\]', '');
else
    line = fgetl(FID);
    line2 = fgetl(FID); % Read the ]
end
if M==1
    transform = str2num(line);
else
    transform(end+1,:) = str2num(line);
end
