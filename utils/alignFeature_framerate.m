function aligned_feat = alignFeature_framerate(feat, framerate, targetFrameRate, targetNFr)
nFr = size(feat,2);
if nFr==1
    aligned_feat = repmat(feat, 1, targetNFr,1);
else
    frame_rate_ratio = targetFrameRate/framerate;
    x_idx = frame_rate_ratio/2 : frame_rate_ratio : (nFr*frame_rate_ratio);    % assume frame shift is half of frame length
    x_idx = x_idx(1:nFr);
    desired_idx = 0.5:(x_idx(end)+frame_rate_ratio*3);
    aligned_feat = interp1(x_idx, feat', desired_idx(1:min(length(desired_idx), targetNFr)), 'nearest', 'extrap')';
end

if 0
    subplot(2,1,1); 
    imagesc(feat);
    subplot(2,1,2);
    imagesc(aligned_feat)'
end
end