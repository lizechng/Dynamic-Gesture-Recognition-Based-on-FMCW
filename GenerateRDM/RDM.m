clear all; clc;
data = readDCA1000('../Data/xz_z_z.bin');
rdm = zeros(256,128);
for cnt = 1:600
    disp(cnt)
    frame = reshape(data(1,256*128*(cnt-1)+1:256*128*(cnt)),256,128);
    rdm = TwoDFFT(frame);
    rdm = abs(rdm);
    rdm = rdm(1:110,40:90);
    imagesc(mag2db(rdm));
    axis('off');
    pause(0.2)
    filename = strcat('XZ/xz-', num2str(cnt));
    saveas(gcf, filename, 'png');
end