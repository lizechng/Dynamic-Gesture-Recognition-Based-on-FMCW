clear all; clc;
data = readDCA1000('F:/Code/GenerateRDM/Data/qh_z_z.bin');
rdm = zeros(256,128);
result = zeros(256,128,200);
for cnt = 1:200
    disp(cnt)
    frame = reshape(data(1,256*128*(cnt-1)+1:256*128*(cnt)),256,128);
    rdm = TwoDFFT(frame);
    rdm = abs(rdm);
    result(:,:,cnt) = rdm;
end
% 
% temp = zeros(16*32,128);
% GenerateDTM
for frame = 1:22
    for i = 1:32
       DTM(16*(i-1)+1:16*i,:) = result(10:25,:,(frame-1)*8+i); 
    end
    imagesc(mag2db(DTM'));
    axis('off');
    pause(0.5);
end

% %% GenerateRTM
% for frame = 1:22
%     for i = 1:32
%        RTM(:,11*(i-1)+1:11*i) = result(:,60:70,(frame-1)*8+i); 
%     end
%     imagesc(mag2db(RTM(1:150,:)));
%     axis('off');
%     pause(0.5);
% end

