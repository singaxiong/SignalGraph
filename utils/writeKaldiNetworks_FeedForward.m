function writeKaldiNetworks_FeedForward(layer, kaldiFileName)

fprintf('Save DNN into Kaldi network format: %s!\n', kaldiFileName);

FID = fopen(kaldiFileName, 'w');
fprintf(FID, '<Nnet>\n');

for i=1:length(layer)
    fprintf('Write layer %d - %s - %s\n', i, layer{i}.name, datestr(now));
    switch lower(layer{i}.name)
        case 'affine'
            [outputSize, inputSize] = size(layer{i}.W);
            fprintf(FID, '<AffineTransform> %d %d\n', outputSize, inputSize);
            fprintf(FID, '<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0  [\n');
            
            for j=1:outputSize-1
                fprintf(FID, '%f ', layer{i}.W(j,:));
                fprintf(FID,'\n');
            end
            fprintf(FID, '%f ', layer{i}.W(end,:));
            fprintf(FID,']\n');
            
            fprintf(FID, '[ ');
            fprintf(FID, '%f ', layer{i}.b);
            fprintf(FID, ']\n');
        case 'sigmoid'
            fprintf(FID, '<Sigmoid> %d %d\n', outputSize, outputSize);
        case 'softmax'
            fprintf(FID, '<Softmax> %d %d\n', outputSize, outputSize);
    end
end
 
fprintf(FID, '</Nnet>\n');
fclose(FID);