
function [mixture, separated, clean, separated_wav, mixtureSTFT, mask] = RunSeparationNN(Data, layer, para)

output = FeatureTree2(Data, para, layer);
for i=1:length(output{1})
    output{1}{i} = gather(output{1}{i});
end

mixtureSTFT = gather(output{1}{1});
mixture = gather(output{1}{2});
separated{1} = gather(output{1}{3});
separated{2} = gather(output{1}{4});

for i=1:length(separated)
    separated_wav{i} = abs2wav(exp(separated{i}(1:257,:)/2)', angle(mixtureSTFT)', 400, 240);
end

if isempty(para.test.clean_idx)
    clean  = [];
else
    for i=1:length(para.test.clean_idx)
        clean{i} = gather(output{1}{para.test.clean_idx(i)});
    end
end

if isempty(para.test.mask_idx)
    mask = [];
else
    for i=1:length(para.test.mask_idx)
        mask{i} = output{1}{para.test.mask_idx(i)};
    end
end

end
