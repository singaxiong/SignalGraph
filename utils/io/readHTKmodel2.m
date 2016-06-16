% read the model file in HTK format
% Author: XIAO XIONG
% Created: 16 Feb, 2004
% Last Modified: 16 Feb, 2004

function models = readHTKmodels(model_file, output_file_name, D_vector)

FILE = fopen( model_file );
N_clusters = 0;
while fsearch('<MEAN>',FILE) == 0
    N_clusters = N_clusters + 1;
    temp = textscan(FILE, '%n', 1);
    mean2(N_clusters,:) = textscan(FILE, '%n', D_vector);
    temp = textscan(FILE,'%s %n',1);
    for i=1:D_vector
        temp = textscan(FILE, '%n', (D_vector-i+1));
        cov2(N_clusters,i,i:D_vector) = temp{1}';
    end
end
cov3 = cell(N_clusters,1);
cov4 = cell(N_clusters,1);
cov5 = cell(N_clusters,1);
for i=1:N_clusters
    for j=1:D_vector
        cov3{i}(j,:) = cov2(i,j,:);
    end
    cov4{i} = cov3{i}+tril(cov3{i}',-1);
    cov5{i} = inv(cov4{i});
end
apriori_weight = ones(N_clusters,1);
writeCluster(D_vector, N_clusters, mean2, cov5, apriori_weight, output_file_name);