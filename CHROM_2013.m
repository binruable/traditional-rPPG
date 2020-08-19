addpath(genpath([cd '\test_data\']))
VideoFile = 'video.avi';  %要预测的视频
FS = 25;                  %视频采样率
StartTime = 0;  %视频开始时间
Duration = 26;  %视频结束时间
LPF = 0.7; %低截止频率
HPF = 2.5; %高截止频率
WinSec=1.6;
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;
FramesToRead=floor(Duration*VidObj.FrameRate); 
T = zeros(FramesToRead,1);
RGB = zeros(FramesToRead,3);
FN = 0;
while hasFrame(VidObj) && (VidObj.CurrentTime <= StartTime+Duration)
    FN = FN+1;
    T(FN) = VidObj.CurrentTime;
    VidFrame = readFrame(VidObj);
    
    %人脸检测
    VidROI = VidFrame;
    SkinSegmentTF = false;
    if(SkinSegmentTF)%皮肤分割（此代码未实现）
        YCBCR = rgb2ycbcr(VidROI);
        Yth = YCBCR(:,:,1)>80;
        CBth = (YCBCR(:,:,2)>77).*(YCBCR(:,:,2)<127);
        CRth = (YCBCR(:,:,3)>133).*(YCBCR(:,:,3)<173);
        ROISkin = VidROI.*repmat(uint8(Yth.*CBth.*CRth),[1,1,3]);
        RGB(FN,:) = squeeze(sum(sum(ROISkin,1),2)./sum(sum(logical(ROISkin),1),2));
    else
        RGB(FN,:) = sum(sum(VidROI,2)) ./ (size(VidROI,1)*size(VidROI,2));
    end
end

NyquistF = 1/2*FS;
[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter - originally specified as an a FIR band-pass filter with cutoff frequencies 40-240 BPM

%50% overlap
WinL = ceil(WinSec*FS);
if(mod(WinL,2))%使窗口大小均匀重叠，增加汉宁加窗信号
    WinL=WinL+1;
end
NWin = floor((FN-WinL/2)/(WinL/2));
S = zeros(NWin,1);
WinS = 1;%开始窗口索引
WinM = WinS+WinL/2;%中间窗口索引
WinE = WinS+WinL-1;%结束窗口索引

for i = 1:NWin
    TWin = T(WinS:WinE,:);
    RGBBase = mean(RGB(WinS:WinE,:));
    RGBNorm = bsxfun(@times,RGB(WinS:WinE,:),1./RGBBase)-1;
    % CHROM
    Xs = squeeze(3*RGBNorm(:,1)-2*RGBNorm(:,2));%论文中式子3Rn-2Gn
    Ys = squeeze(1.5*RGBNorm(:,1)+RGBNorm(:,2)-1.5*RGBNorm(:,3));%1.5Rn+Gn-1.5Bn
    Xf = filtfilt(B,A,double(Xs));
    Yf = filtfilt(B,A,double(Ys));
    Alpha = std(Xf)./std(Yf);
    SWin = Xf - Alpha.*Yf;
    SWin = hann(WinL).*SWin;
    %汉明窗重叠
    if(i==1)
        S = SWin;
        TX = TWin;
    else
        S(WinS:WinM-1) = S(WinS:WinM-1)+SWin(1:WinL/2);
        S(WinM:WinE) = SWin(WinL/2+1:end);
        TX(WinM:WinE) = TWin(WinL/2+1:end);
    end
    WinS = WinM;
    WinM = WinS+WinL/2;
    WinE = WinS+WinL-1;
end
BVP=S;

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
