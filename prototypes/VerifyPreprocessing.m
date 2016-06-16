function [processing] = VerifyPreprocessing(visible, reader, preprocessing, nUttUsed)
if exist('nUttUsed')==0 || length(nUttUsed)==0
    nUttUsed = 500;
end

nUtt = length(visible.data);
if nUtt>nUttUsed
    step = round(nUtt/nUttUsed);
    data = visible.data(1:step:end);
else
    data = visible.data;
end

if visible.isFileName
    for i=1:length(data)
        feat{i} = InputReader(data{i}, reader, 0, 0);
    end
else
    feat = data;
end

for i=1:length(feat)
    feat{i} = FeaturePipe(feat{i}, preprocessing);
end
feat = cell2mat(feat);

plot(mean(feat,2)); hold on;
plot(std(feat')); hold off
end
