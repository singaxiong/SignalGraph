
%%%@function: TDOA estimation using GCC-PHAT%%%%%%%%%%
function [tde_est_Gcc]=TimeDelayEstimation(sig, nChs)
Ref_Ch = 1;

frame_sz = length(sig(:,1)); %64ms The incremental size for DOA
Window_Sz = frame_sz; %256ms The window size for DOA
buf = zeros(Window_Sz,nChs);

nFrames = floor(length(sig(:,1))/frame_sz);
%%Calculate the time delays between Mic 1 and Mic 2, 3, ...,8.
TDE_GCC=[];
for iframe=1:nFrames
    tde_est = zeros(1,nChs-1);
    st = (iframe-1)*frame_sz+1;
    et = iframe*frame_sz;
    ref_ch_frame = sig(st:et,Ref_Ch);
    buf(:,1) = [buf(frame_sz+1:end,1);ref_ch_frame];
    fftx=fft(buf(:,1).*hamming(Window_Sz));
    tde_count = 1;
    for Ref_Ch=1
        for ich=Ref_Ch+1:nChs
            if ich~=Ref_Ch
                tmp_frame = sig(st:et,ich);
                buf(:,ich) = [buf(frame_sz+1:end,ich);tmp_frame];
                ffty=fft(buf(:,ich).*hamming(Window_Sz));
                Pxy=fftx.*conj(ffty);  % compute the cross-spectrum
                weight_factor=max(abs(Pxy),1e-6); %create the weight factor
                norm_Pxy=Pxy./weight_factor;
                r_GCC=fftshift(real(ifft(norm_Pxy)),1);
                r_GCC=flipud(r_GCC);
                [val, index]=max(r_GCC);
                while abs(index-Window_Sz/2)>10
                    [val, index]=max(r_GCC);
                    r_GCC(index)=0;
                end
                % figure(10);plot(r_GCC); hold off;
                % tde_est(tde_count)=index-Window_Sz/2;
                TDE_GCC(iframe,tde_count) = floor(index-Window_Sz/2);
                tde_count = tde_count + 1;
                
            end
        end
    end
end
%         figure(1);plot(TDE_vect,'*');
tde_est_Gcc = mode(TDE_GCC,1);
tde_est_Gcc = [0 tde_est_Gcc];