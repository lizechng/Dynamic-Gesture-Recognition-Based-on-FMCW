%% 2D-FFT
function rdm = TwoDFFT(frame)
range_win = hamming(256);   %加海明窗
doppler_win = hamming(128);

for m=1:128
  temp=frame(:,m).*range_win;    %加窗函数
  temp_fft=fft(temp);    %对每个chirp做N点FFT
  rdm(:,m)=temp_fft;
end

for n=1:256
  temp=rdm(n,:).*(doppler_win)';    
  % temp=range_profile(n,:,k);
  temp_fft=fftshift(fft(temp));    %对rangeFFT结果进行M点FFT
  rdm(n,:)=temp_fft;  
end
rdm = rdm;