VideoFile = 'video.avi';  %要预测的视频
FS = 25;                  %视频采样率
LPF = 0.7; %低截止频率
HPF = 2.5; %高截止频率
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;  
FramesToRead=ceil(Duration*VidObj.FrameRate); 
T = zeros(FramesToRead,1);
RGB = zeros(FramesToRead,3);
FN = 0; 
while hasFrame(VidObj) && (VidObj.CurrentTime <= StartTime+Duration)
    FN = FN+1;
    T(FN) = VidObj.CurrentTime;
    VidFrame = readFrame(VidObj);
    VidROI = VidFrame;
    RGB(FN,:) = sum(sum(VidROI));
end
BVP = RGB(:,2);  %提取整个视频的绿色通道
NyquistF = 1/2*FS;
[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter - originally specified in reference with a 4th order butterworth using filtfilt function
BVP_F = filtfilt(B,A,(double(BVP)-mean(BVP)));
BVP = BVP_F;  %过滤之后的绿色通道

LL_PR = 40;  %最低bpm
UL_PR = 200; %最高bpm
Nyquist = FS/2;
FResBPM = 0.5; 

N = (60*2*Nyquist)/FResBPM;

% 估计功率谱密度（PSD）
[Pxx,F] = periodogram(BVP,hamming(length(BVP)),N,FS);
FMask = (F >= (LL_PR/60))&(F <= (UL_PR/60));

FRange = F(FMask);
PRange = Pxx(FMask);
MaxInd = argmax(Pxx(FMask),1);
PR_F = FRange(MaxInd);
HR = PR_F*60;   %预测出来的心率
