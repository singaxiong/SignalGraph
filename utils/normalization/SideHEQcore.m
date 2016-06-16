function [mfc] = SideHEQcore(mfcRaw, RefFile)
addpath('F:\Dropbox\Workspace\Projects\Normalization\PHEQ');
RefFile = 'F:\Dropbox\Workspace\Projects\Normalization\PHEQ\Data/SHEQ.Gaussian.O10.N100000';
para = readHEQstats(RefFile);
para.tied = 0;

mfcRaw2 = cell2mat(mfcRaw');

mfcHEQ = HEQp_general(mfcRaw2, para);

for j=1:length(mfcRaw)
    mfcMVN{j} = bsxfun(@minus, mfcRaw{j}, meanV);
    mfcMVN{j} = bsxfun(@times, mfcMVN{j}, precision);
end
