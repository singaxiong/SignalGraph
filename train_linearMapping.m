function [processing, cost, recon_t, cost_t]         = train_linearMapping(visible_train, target_train, visible_test, target_test, para)
if isfield(para, 'notest') && para.notest==1
    notest = 1; else    notest = 0; end
if isfield(para, 'biasonly')
    biasonly = para.biasonly; else    biasonly = 0; end
if isfield(para, 'L2weight')
    L2weight = para.L2weight; else    L2weight = 0; end

[dim_v, nSample_v] = size(visible_train);
[dim_t, nSample_t] = size(visible_train);
if nSample_v~=nSample_t
    fprintf('Error: the number of training samples in visible and target are different\n'); return;
end

if dim_v==dim_t && biasonly>0
    A = eye(dim_v);
    b = mean(target_train-visible_train,2);
    if biasonly==2
        b = ones(dim_v,1) * mean(b);
    end
else
    visible_train2 = CMN(visible_train')';
    target_train2 = CMN(target_train')';
    R = visible_train2 * visible_train2' / nSample_v;
    p = visible_train2*target_train2' / nSample_v;
    A = inv( R + eye(dim_v)*L2weight ) * p;
    b = mean(target_train,2) - A' * mean(visible_train,2);
end

processing{1}.name = 'AffineTransform';
processing{end}.transform = A';
processing{end}.bias = b;

recon = FeaturePipe(visible_train, processing);
cost = 0.5*sum(sum( (recon-target_train).^2 )) / size(visible_train,2);

if notest
    nDisply = min(nSample_v,1000);
    recon_t = [];
    cost_t = [];
    imagesc([target_train(:,1:nDisply); ones(1,nDisply); recon(:,1:nDisply)]);
else
    nDisply = min(size(visible_test,2),1000);
    recon_t = FeaturePipe(visible_test, processing);
    cost_t = 0.5*sum(sum( (recon_t-target_test).^2 )) / size(visible_test,2);
    imagesc([target_test(:,1:nDisply); ones(1,nDisply); recon_t(:,1:nDisply)]); 
end
end
