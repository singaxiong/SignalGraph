function score = readHTKmlf(file_name)


FID = fopen(file_name,'r');

mask = {'sil','sp','one','two','three','four','five','six','seven','eight'...
    'nine','oh','zero'};

cnt = 1;
tmp = textscan(FID,'%s',1);
while 1
    tmp = textscan(FID,'%s',1);
    if(isempty(tmp{1})) % if it is the end of the file
        textscan(FID,'%s',1);
        break; 
    end
    utt_name = tmp{1}{1}(4:length(tmp{1}{1})-5);
    score.name{cnt} = utt_name;
    score.(utt_name) = struct;

    for i=1:13  % initialize the model count
        model_cnt.(mask{i}) = 0;
    end
    while 1
        tmp = textscan(FID,'%d %d %s %f %s',1);
        if(isempty(tmp{1})) %if it is the end of a model
            textscan(FID,'%s',1);
            break; 
        end

        idx1 = findstr(tmp{3}{1},'[');
        idx2 = findstr(tmp{3}{1},']');
        state = str2num( tmp{3}{1}(idx1+1:idx2-1) );
        if state==2 % if it is the start of a model
            % if there is multiple identical model, number them as model1,
            % model2,...
            model = tmp{3}{1}(1:idx1-1);    
            model_cnt.(model) = model_cnt.(model) + 1;
            model = sprintf('%s%d',model,model_cnt.(model));
        end
        state = str2num( tmp{3}{1}(idx1+1:idx2-1) );    %state number
        score.(utt_name).(model)(state-1,1) = double(tmp{1})/10^5;  %start frame
        score.(utt_name).(model)(state-1,2) = double(tmp{2})/10^5;  %stop frame
        score.(utt_name).(model)(state-1,3) = tmp{4};   % state likelihood
    end
    cnt = cnt + 1;
end
fclose(FID);