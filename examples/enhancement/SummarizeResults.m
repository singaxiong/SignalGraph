function SummarizeResults()

files = findFiles('nnet/scores');


for i=1:length(files)
    score = load(files{i});
    curr_scores = score.scores;
    [nMeasures, nSys] = size(score.scores{1});
    S = cell2mat_tensor3D(curr_scores);
    avgS = mean(S,3);
    fprintf('%s:\t\t\t', files{i})
    for j=1%:nMeasures
        fprintf('%s: ', score.measures{j});
        fprintf('\t%2.2f ', avgS(j,:));
        fprintf('\n');
        if strcmpi(score.measures{j}, 'pesq')
            PESQ(i,:) = avgS(j,:);
        end
    end
    
    
end

mean(PESQ)



end
