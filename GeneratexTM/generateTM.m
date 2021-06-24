%function [X, Y1, Y2] = generateTM(data)

maxFrame = 256;
AdcSample = 256;
PulsePerFrame = 128;

%% Define Variables
d = zeros(AdcSample,PulsePerFrame,maxFrame);
range_fft = zeros(AdcSample,PulsePerFrame);
doppler_fft = zeros(AdcSample,PulsePerFrame);
X = zeros(AdcSample,PulsePerFrame);
Y1 = zeros(AdcSample,PulsePerFrame);
Y2 = zeros(AdcSample,PulsePerFrame);

for i = 1:maxFrame
    d(:,:,i) = reshape(data(1,AdcSample*PulsePerFrame*(i-1)+1:AdcSample*PulsePerFrame*i),AdcSample,PulsePerFrame);
end
demo = d(:,:,1);

%% Fix Range
for i = 1:PulsePerFrame
    range_fft(:,i) = (fft(demo(:,i)));
end
demo_range = abs(fftshift(range_fft));
max_range = find(sum(demo_range,2) == max(sum(demo_range,2)));

%% Fix Doppler
for i = 1:AdcSample
    doppler_fft(i,:) = fft(range_fft(i,:));
end
demo_doppler = abs(fftshift(doppler_fft));
max_doppler = find(sum(demo_doppler,1) == max(sum(demo_doppler,1)));

%% DTM
for i = 1:maxFrame
    val = d(:,:,i);
    for j = 1:PulsePerFrame
        val(:,j) = fft(val(:,j));
    end
    % w/o hamming
    tmp = fftshift(fft(val(130,:)));
    X(i,:) = tmp;
end

%% RTM_1
for i = 1:maxFrame/2
    row = d(:,:,i);
    tmp = fftshift(fft(row(:,max_doppler)));
    Y1(:,i) = tmp;
end
%% RTM_2
for i = maxFrame/2+1:maxFrame
    row = d(:,:,i);
    tmp = fftshift(fft(row(:,max_doppler)));
    Y2(:,i-maxFrame/2) = tmp;
end
        