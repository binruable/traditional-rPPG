addpath(genpath([cd '\test_data\']))
addpath(genpath([cd '\tools\']))
VideoFile = 'video.avi';  %要预测的视频
FS = 25;                  %视频采样率
StartTime = 0;  %视频开始时间
Duration = 26;  %视频结束时间
LPF = 0.7; %低截止频率
HPF = 2.5; %高截止频率
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;
FramesToRead=ceil(Duration*VidObj.FrameRate); 
T = zeros(FramesToRead,1);%初始化时间序列
RGB=zeros(FramesToRead,3);%初始化颜色通道
FN=0;
while hasFrame(VidObj) && (VidObj.CurrentTime <= StartTime+Duration)
    FN = FN+1;
    T(FN) = VidObj.CurrentTime;
    VidFrame = readFrame(VidObj);
    VidROI = VidFrame; 
    RGB(FN,:) = mean(sum(VidROI));
end

% 去趋势化 和 ICA
NyquistF = 1/2*FS;
RGBNorm=zeros(size(RGB));
Lambda=100;
for c=1:3
    T=length(RGB(:,c));
    I=speye(T);
    D2=spdiags(ones(T-2,1)*[1 -2 1],[0:2],T-2,T);
    sr=double(RGB(:,c));
    y=(I-inv(I+Lambda^2*(D2'*D2)))*sr;
    RGBNorm(:,c) = (RGBDetrend - mean(RGBDetrend))/std(RGBDetrend);%归一化
end
[nRows, nCols] = size(RGBNorm');
if nRows > nCols
    error('行不能比列大！');
end
Nsources = 3;
if Nsources > min([nRows nCols])
    Nsources = min([nRows nCols]);
end
[Winv, Zhat] = jade(RGBNorm',Nsources); 
W = pinv(Winv);
S = Zhat;
MaxPx=zeros(1,3);
for c=1:3
    FF = fft(S(c,:));
    F=(1:length(FF))/length(FF)*FS*60;
    FF(1)=[];
    N=length(FF);
    Px = abs(FF(1:floor(N/2))).^2;
    Fx = (1:N/2)/(N/2)*NyquistF;
    Px=Px/sum(Px);
    MaxPx(c)=max(Px);
end
[M,MaxComp]=max(MaxPx(:));
BVP_I = S(MaxComp,:);
[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%三阶带通滤波
BVP_F = filtfilt(B,A,double(BVP_I));
BVP=BVP_F;

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
[~,MaxInd] = max(Pxx(FMask),[],1);
PR_F = FRange(MaxInd);
HR = PR_F*60;   %预测出来的心率