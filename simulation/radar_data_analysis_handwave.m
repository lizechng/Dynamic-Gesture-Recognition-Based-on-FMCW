%%
rx1_c = I1 + Q1*1j;
rx2_c = I2 + Q2*1j;
[total_chirps,chirp_data_len] = size(rx1_c);

%%
disp('Calibrating chirp signal amptitude...')
for k=1:total_chirps
    for m=1:chirp_data_len
        rx1_c(k,m) = rx1_c(k,m)/((m+1)/chirp_data_len)^0.3;
        rx2_c(k,m) = rx2_c(k,m)/((m+1)/chirp_data_len)^0.3;
    end
end
rx1_c = rx1_c(:,300:1324);
rx2_c = rx2_c(:,300:1324);
[total_chirps,chirp_data_len] = size(rx1_c);
disp('Finished calibrating chirp signal amptitude.')

%%
mx_rx = rx1_c.*conj(rx2_c);
% Low Pass in-chirp filter
lp_mx = zeros(size(mx_rx));
for chirp_idx = 1:size(lp_mx,1)
    lp_mx(chirp_idx,:) = lp_filter(mx_rx(chirp_idx,:));
end

%%

RANGE_LIMIT_MIN = 4;
RANGE_LIMIT_MAX = 16;
RANGE_FFT_CLIP_SIZE = RANGE_LIMIT_MAX;

DPL_FFT_SIZE = 64; 
DPL_FFT_INTERVAL = 8;
DPL_MAP_CLIP_RSIZE = RANGE_LIMIT_MAX;
DPL_MAP_CLIP_VSIZE = DPL_FFT_SIZE-4;

%%

% Range FFT %
disp('Calculating Range FFT...')
rx1_fft = matrix_fft_2ndD(rx1_c);
rx2_fft = matrix_fft_2ndD(rx2_c);

% Range Band
rx1_fft(:,1:RANGE_LIMIT_MIN-1) = rx1_fft(:,1:RANGE_LIMIT_MIN-1)*1e-10;
rx1_fft(:,RANGE_LIMIT_MAX:chirp_data_len) = rx1_fft(:,RANGE_LIMIT_MAX:chirp_data_len)*1e-10;
rx2_fft(:,1:RANGE_LIMIT_MIN-1) = rx2_fft(:,1:RANGE_LIMIT_MIN-1)*1e-10;
rx2_fft(:,RANGE_LIMIT_MAX:chirp_data_len) = rx2_fft(:,RANGE_LIMIT_MAX:chirp_data_len)*1e-10;

% Clip Range
rx1_fft = rx1_fft(:,1:RANGE_FFT_CLIP_SIZE);
rx2_fft = rx2_fft(:,1:RANGE_FFT_CLIP_SIZE);

% Mix Range
mx_range_fft = rx1_fft.*conj(rx2_fft);

disp('Finished Range FFT caculation.')

%%
Rr = zeros(1,fix(total_chirps/DPL_FFT_INTERVAL+1));
Vr = zeros(1,fix(total_chirps/DPL_FFT_INTERVAL+1));
Rt = zeros(1,fix(total_chirps/DPL_FFT_INTERVAL+1));
Vt = zeros(1,fix(total_chirps/DPL_FFT_INTERVAL+1));
dpl_frame_idx = 0;

for frame_start_chirp_idx = 1:DPL_FFT_INTERVAL:total_chirps-DPL_FFT_SIZE

    dpl_frame_idx = dpl_frame_idx+1;

    range_rx1_fft = rx1_fft(frame_start_chirp_idx:frame_start_chirp_idx+DPL_FFT_SIZE-1,1:RANGE_FFT_CLIP_SIZE);
    range_rx2_fft = rx2_fft(frame_start_chirp_idx:frame_start_chirp_idx+DPL_FFT_SIZE-1,1:RANGE_FFT_CLIP_SIZE);
    drange_fft = mx_range_fft(frame_start_chirp_idx:frame_start_chirp_idx+DPL_FFT_SIZE-1,1:RANGE_FFT_CLIP_SIZE);

    % Vr Dopplor FFT %
    vr1_dpl_fft = matrix_fft_2ndD_shift(range_rx1_fft')';
    vr2_dpl_fft = matrix_fft_2ndD_shift(range_rx2_fft')';
    vr1_dpl_fft(DPL_FFT_SIZE/2+1,:) = 0;
    vr2_dpl_fft(DPL_FFT_SIZE/2+1,:) = 0;
    % Vt Dopplor FFT
    vt_dpl_fft = matrix_fft_2ndD_shift(drange_fft')';
    vt_dpl_fft(DPL_FFT_SIZE/2+1,:) = 0;

    v_clip_idx_s = DPL_FFT_SIZE/2+1 - DPL_MAP_CLIP_VSIZE/2;
    v_clip_idx_e = DPL_FFT_SIZE/2+1 + DPL_MAP_CLIP_VSIZE/2;
    vr1_dpl_fft_c = vr1_dpl_fft(v_clip_idx_s:v_clip_idx_e,1:DPL_MAP_CLIP_RSIZE);
    vr2_dpl_fft_c = vr2_dpl_fft(v_clip_idx_s:v_clip_idx_e,1:DPL_MAP_CLIP_RSIZE);
    vt_dpl_fft_c = vt_dpl_fft(v_clip_idx_s:v_clip_idx_e,1:DPL_MAP_CLIP_RSIZE);

    % R-V energy
    eng_vr1_avg = (avg_filter_2D(abs(vr1_dpl_fft_c),2)*1e-5);
    eng_vr2_avg = (avg_filter_2D(abs(vr2_dpl_fft_c),2)*1e-5);
    eng_vt_avg = (avg_filter_2D(abs(vt_dpl_fft_c),2)*1e-10);
    
% DEBUG ONLY:
% max(max(eng_vr1_avg))
% max(max(eng_vr2_avg))
% max(max(eng_vt_avg))
% pause(1)

    eng_vr_mx = eng_vr1_avg.*eng_vr2_avg;
    eng_vr_mx(eng_vr_mx<1) = 0;
    eng_vr_mx = avg_filter_2D(eng_vr_mx,2);
    threshold_eng_vr = max(max(eng_vr_mx))*0.015 + 0.1*((DPL_FFT_SIZE/64) + 1);
    eng_vr_mx(eng_vr_mx<threshold_eng_vr) = 0;

    eng_vt_avg(eng_vt_avg<1) = 0;
    eng_vt_avg = avg_filter_2D(eng_vt_avg,2);
    threshold_eng_vt = max(max(eng_vt_avg))*0.15 + 0.1*((DPL_FFT_SIZE/64) + 1);
    eng_vt_avg(eng_vt_avg<threshold_eng_vt) = 0;

    % Count blocks
    %   - find and countnumber of signal blocks
    %   - in current application, only one target object is concerned 
    %   - consider to be noise (no target object) when more than 1 blocks
    [eng_vr_bin,eng_vr_num] = erode_tobin_2D(eng_vr_mx);
    [eng_vt_bin,eng_vt_num] = erode_tobin_2D(eng_vt_avg);

    % ************** Rr-Vr
    if eng_vr_num==1
        [Rr(dpl_frame_idx),Vr_grav] = center_gravity_2D(eng_vr_mx);
        Vr(dpl_frame_idx) = Vr_grav - DPL_MAP_CLIP_VSIZE/2 - 1;
    else
        eng_vr_mx = zeros(size(eng_vr_mx));
        eng_vr_bin = zeros(size(eng_vr_bin));
        Rr(dpl_frame_idx) = -inf;
        Vr(dpl_frame_idx) = -inf;
    end

% DEBUG ONLY:
mesh(eng_vt_avg);pause(0.02);
% dpl_frame_idx
% max(max(eng_vt_avg)) 

    % ************** Rt-Vt
    if eng_vt_num==1 || eng_vt_num==2
        [Rt(dpl_frame_idx),Vt_grav] = center_gravity_2D(eng_vt_avg);
        Vt(dpl_frame_idx) = Vt_grav - DPL_MAP_CLIP_VSIZE/2 - 1;
        Vt(dpl_frame_idx) = Vt(dpl_frame_idx) /abs(max(Vt(dpl_frame_idx))) *(max(max(eng_vt_avg))^2);
    else
        eng_vt_avg = zeros(size(eng_vt_avg));
        eng_vt_bin = zeros(size(eng_vt_bin));
        Rt(dpl_frame_idx) = -inf;
        Vt(dpl_frame_idx) = -inf;
    end

    if mod(dpl_frame_idx,100)==0
        disp(['Calculating Doppler FFT... ', num2str(dpl_frame_idx),'/',num2str(fix(size(Rr,2)/100)*100)])
    end
end
disp('Finished Doppler FFT.')

Rr = Rr(1:dpl_frame_idx);
Vr = Vr(1:dpl_frame_idx);
Rt = Rt(1:dpl_frame_idx);
Vt = Vt(1:dpl_frame_idx);
Zo = zeros(1,dpl_frame_idx);

%%
FFT_RESOLUTION = 1024;

phi = zeros(1,fix(total_chirps/DPL_FFT_INTERVAL+1));
phi_sum = zeros(1,fix(total_chirps/DPL_FFT_INTERVAL+1));


for chirp_cut_time = 300:600 
    if mod(chirp_cut_time,100)==0
        disp(['Calculating W...',num2str(chirp_cut_time),'/?'])
    end

    ts_cut_mx = lp_mx(1:total_chirps,chirp_cut_time);
    ts_cut_mx = ts_cut_mx - mean(ts_cut_mx);
    fft_idx = 0;

    for chirp_idx = 1:DPL_FFT_INTERVAL:total_chirps-DPL_FFT_SIZE
        fft_idx = fft_idx+1;

        ts_sample = ts_cut_mx(chirp_idx:chirp_idx+DPL_FFT_SIZE-1);
        ts_sample(FFT_RESOLUTION)=0;
        ts_fft = fft(ts_sample);
        ts_fft(1) = 0;
        ts_ffts = fftshift(ts_fft);    
        ts_ffts_abs = abs(ts_ffts);

        ts_fft(ts_ffts_abs<1e8)=0;
        ts_ffts(ts_fft==0)=0;
        ts_ffts_abs(ts_fft==0)=0;

        phi_cur = find(ts_ffts_abs==max(ts_ffts_abs));
        phi_cur=phi_cur(1);
        if phi_cur==1
            phi_cur=FFT_RESOLUTION/2 + 1;
        end
        phi(fft_idx) = phi_cur - FFT_RESOLUTION/2 - 1;
    end
    phi_sum(1:fft_idx) = phi_sum(1:fft_idx)+phi(1:fft_idx);
end
phi_sum = phi_sum(1:fft_idx)/1e4;
phi_sum = phi_sum - mean(phi_sum);
W = -phi_sum;

disp('Finished calculating W.')
%%
figure

subplot(3,1,1)
plot(Zo,':g');hold on
h1 = plot(Rr,'r');
h2 = plot(Vr,'b');
h3 = plot(Rt,'c');
legend([h1,h2,h3],'Rr','Vr','Rt')
% xlabel('Time Index')
ylabel('Rr , Vr , Rt')
set(gca,'yticklabel',[])
hold off

subplot(3,1,2)
plot(Zo,':g');hold on
h4 = plot(Vt,'b');
legend(h4,'Vt')
%     xlabel('Time Index')
ylabel('Left - Right')
set(gca,'yticklabel',[])
hold off

subplot(3,1,3)
%     plot(Zo,':g');hold on
h5 = plot(W,'b');
legend(h5,'W')
xlabel('Time Index')
ylabel('Left - Right')
set(gca,'yticklabel',[])
hold off

pause(0)
