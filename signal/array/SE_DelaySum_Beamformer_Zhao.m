function main()
clear all;

addpath functions

root = 'C:\SK-workspace\myCodes\robust_beamformer_oxyz\Reverber_Challenge2013\Audio\Evaluation_Data/';

%%%%number of microphones from 2 & 8 %%%%%%%%%%%%
for nChs = [2 8]  %%% = 2 or 8
    %%select one room to process%%%%
    for ROOM_TYPE = [1:8]  %%choose 1,2,3,4,5,6,7,8
        %%%initialize the channel parameters
        iniPara;
        
        if ROOM_TYPE==1||ROOM_TYPE==2
            in_audio_root = [root data_type{1}];
            out_audio_root = [root data_type{1} '_' num2str(nChs) 'ch_DS_out'];
            
        end
        
        if ROOM_TYPE>=3&&ROOM_TYPE<=8
            in_audio_root = [root data_type{2} '/data'];
            out_audio_root = [root data_type{2} '_' num2str(nChs) 'ch_DS_out/data'];
            
        end
        
        
        task_file_root = [root 'taskFiles_et'];
        
        %%%%%specify the number of channels (8 is used)%%%%%%
        
        Chs = {'A','B','C','D','E','F','G','H'};
        for ich = 1:nChs
            in_dir_txt{ich} = [task_file_root '/' num2str(nChs) 'ch/' rm_type{ROOM_TYPE} '_' Chs{ich}];
        end
        
        rm_type_str = strrep(rm_type{ROOM_TYPE}, [num2str(nChs) 'ch'], '1ch');
        out_dir_txt = [task_file_root '/1ch/' rm_type_str '_A'];
        dFs = 16000;
        File_counter = 0;
        
        for ich = 1:nChs
            fid(ich) = fopen(in_dir_txt{ich});
        end
        fid_out = fopen(out_dir_txt);
        tline = fgetl(fid_out);
        file_path_out = [out_audio_root tline];
        while ischar(tline)
            File_counter = File_counter + 1
            disp(file_path_out)
            for ich = 1:nChs
                tline = fgetl(fid(ich));
                file_path_in = [in_audio_root tline];
                [tmp_sig dFs] = wavread(file_path_in);
                if ich==1
                    sig = zeros(length(tmp_sig),nChs);
                    sig(:,ich) = tmp_sig;
                else
                    sig(:,ich) = tmp_sig;
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%Call TDOA estimator%%%%%%%%%%%%%
            [tde_est_Gcc] = TimeDelayEstimation(sig, nChs);
            %%%%Call Delay and Sum beamformer%%%%
            [out] = DelaySum_Beamformer_fast(sig,tde_est_Gcc);  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             fileWrite(out, dFs, file_path_out);
             
            tline = fgetl(fid_out);
            file_path_out = [out_audio_root tline];
        end
        for ich = 1:nChs
            fclose(fid(ich));
        end
        fclose(fid_out);
    end
end

%%%%@write to file
function fileWrite(xout, dFs, FilePath_out)
idx = strfind(FilePath_out,'/');
out_file_path = FilePath_out(1:idx(end));
folder_exist = exist(out_file_path,'dir');
if ~folder_exist
    mkdir(out_file_path);
end
% wavwrite(xout,dFs,FilePath_out);
wavwrite(xout/max(xout)*0.5,dFs,FilePath_out);

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
tde_est_Gcc = [0 tde_est_Gcc]

%%%%%%@function: Delay and Sum beamformer%%%%%%%%%%%%
function [xout] = DelaySum_Beamformer_fast(sig,tde_est)
%%start to process with beamformer
nChs = size(sig,2);

%%perform Delay and Sum beamforming
lfft = 1024; %32ms window size for beamforming
ShiftP = 0.25;
Wsz = lfft;
INC = Wsz*ShiftP;
W = hann(Wsz);
%Apply the Pre-emphasis
pre_emph=0.925;
sig=filter([1 -pre_emph],1,sig);
%%segment the signals into frames
for ich = 1:nChs
    Seg_ch(:,ich,:) = enframe(sig(:,ich),W, INC).';
end
% outY = zeros(size(Seg_ch(:,:,1)));
fft_Seg_ch = fft(Seg_ch,lfft);
% hfft_Seg_ch = fft_Seg_ch(1:lfft/2,:,:);
% mi = 1:lfft/2;
mi = 1:lfft;

% hfft_Seg_ch = fft_Seg_ch;

mi = mi(:);
for ich = 1:nChs
    EV(:,ich) = exp(-1i*2*pi*mi./lfft*tde_est(ich));
end
mat_EV = repmat(EV,[1 1 size(fft_Seg_ch,3)]);
% newnFrame = size(outY,1);
mat_Y = conj(mat_EV).*fft_Seg_ch;

outY = reshape(sum(mat_Y,2),size(fft_Seg_ch,1), size(fft_Seg_ch,3));

% outY = [houtY;flipud(houtY)];
% xoutnow = ifft(outY);
xoutnow = ifft(outY);
xoutnow = xoutnow.';
xout = overlapadd(xoutnow,W,INC);
xout = real(xout);
%Undo the effect of Pre-emphasis
xout=filter(1,[1 -pre_emph],xout);