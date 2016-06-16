%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%@function: Calculate the steered response power along each direction angle and frame.
%%%This method is based on J.H. Dibiase's PhD thesis at p93.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [power_matrix_ang_frame]=getPowerMatrix(input, winSize, overlap, st_vect, useGPU)

%%input:   input signals with dimesion M x N, where M is the number of
%%%        channels, and N is the total length of signal
%%winSize: the window size used for computing the correlation vector
%%overlap: the percentage of overlap between windows
%%st_vect: the steering vector matrix (frequency, angle, channle)
%%power_matrix_ang_frame: output matrix for the steered power at (angle,
%%frame)

if useGPU
    input = gpuArray(input);
end

M = size(input,1);%the number of channels in the array
N = size(input,2);%%total length of input signal
lfft = winSize;
nWindowFrames = floor((N-winSize)/(winSize*(1-overlap))+1); %%compute the total window frames in the input signals
if useGPU
    power_matrix_ang_frame = gpuArray.zeros(360,nWindowFrames);
else
    power_matrix_ang_frame = zeros(360,nWindowFrames);
end
for iframe=1:nWindowFrames
    if iframe==1
        st = 1;
        et = winSize;
        window_Segs = input(:,st:et)';
    else
        st = (iframe-1)*(winSize*(1-overlap))+1;
        et = st + winSize-1;
        window_Segs = input(:,st:et)';
    end
    
    ham_wind = repmat(hamming(winSize), 1,M);
    fftX=fft(window_Segs(:,:).*ham_wind);
    if useGPU
        norm_fftX = gpuArray.zeros(size(fftX));
    else
        norm_fftX = zeros(size(fftX));
    end
    for m = 1:M
        weight_factor = max(abs(fftX(:,m)),1e-6); %create the weight factor
        norm_fftX(:,m)=fftX(:,m)./weight_factor;
    end
    
    if 0
        for bin=1:lfft
            st_v = st_vect(bin,:,:);
            st_v = reshape(st_v,360,M);
            Y = conj(st_v)*norm_fftX(bin,:).';  %%Y(Theta) = sum_m=1:M_Xm/|Xm|*e^(jwtau)
            power_matrix_ang_frame(:,iframe) = power_matrix_ang_frame(:,iframe)+ abs(Y.*conj(Y));
        end
    else
        power_i = 0;
        for m=1:M
            tmp = bsxfun(@times, conj(st_vect(:,:,m)), norm_fftX(:,m));
            power_i = power_i + tmp;
        end
        power_matrix_ang_frame(:,iframe) = sum(abs(power_i.*conj(power_i)));
    end
end

power_matrix_ang_frame = gather(power_matrix_ang_frame);

end