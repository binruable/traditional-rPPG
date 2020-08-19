addpath(genpath([cd '\test_data\']))
VideoFile = 'video.avi'; 
LPF = 0.7; 
HPF = 2.5; 
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;
FramesToRead=ceil(Duration*VidObj.FrameRate);
BBox = [1,1,VidObj.width,VidObj.height];
MinQ = 0.01;
VidFrame = readFrame(VidObj);
try
    Points0 = detectMinEigenFeatures(rgb2gray(VidFrame),'ROI',BBox,'MinQuality',MinQ);
    if length(Points0) < 5
        BCG=NaN; PR=NaN; HR_ECG=NaN; PR_PPG=NaN; SNR=NaN;
        return
    end
    tracker = vision.PointTracker;
    initialize(tracker,Points0.Location,VidFrame);
    T = zeros(FramesToRead-1,1);%Initialize time vector
    Y = zeros(FramesToRead-1,size(Points0.Location,1));%initialize Y
    FN = 0;
    while hasFrame(VidObj) && (VidObj.CurrentTime <= StartTime+Duration)
        FN = FN+1;
        T(FN) = VidObj.CurrentTime;
        VidFrame = readFrame(VidObj);
        %ROIÇøÓò¼ì²â(ÈËÁ³¼ì²â)
        [Points, Validity] = step(tracker,VidFrame);
        Y(FN,:) = Points(:,2);
    end
catch TrackErrorter
    if(strcmp(TrackError.message,'Undefined function ''detectMinEigenFeatures'' for input arguments of type ''uint8''.'))
        [VideoFilePath,VideoFileName,VideoFileExt] = fileparts(VideoFile);
        if(strcmp([VideoFileName VideoFileExt],'video.avi'))
            if(exist(VideoFile,'file'))
                fprintf('Tracking could not be completed without the Computer Vision Toolbox.\nLoading previously run tracking results for ''video_example.mp4''.\n')
                load([VideoFilePath '\BCGTracking.mat']);%contains Y and T results for video_example.mp4 run with defaults FS=30, StartTime=0, and Duration=60
            end
        else
            TrackError
        end
    else
        TrackError
    end
end
MaxYMotions=max(floor(diff(Y,1,2)));
UnstableMask=MaxYMotions>mode(MaxYMotions);
YS=Y(:,~UnstableMask);
NyquistF = 1/2*FS;
[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);
Y_Filt = filtfilt(B,A,double(YS));

Y_Filt2 = bsxfun(@minus, Y_Filt, Y_Filt(1,:));
YL2 = sqrt(sum(Y_Filt2.^2,2));
YMask = YL2<=quantile(YL2,.75);

Y_Filt3=Y_Filt2(YMask,:);
[Coeff, ~, Latent] = pca(Y_Filt3);

Score = bsxfun(@minus, Y_Filt, mean(Y_Filt))/Coeff';

F=find(T>0);
[Pxx,F2] = plomb(Score(F,1:5),T(F));

FMask = (F2 >= LPF)&(F2 <= HPF);
FRange = F2(FMask);
R = zeros(1,5);
for i = 1:5
    [MaxP,IDXP] = max(Pxx(:,i));
    R(i) = (MaxP+Pxx(IDXP*2,i))/sum(Pxx(:,i));
end
[~,NumPC] = max(R);

HR = FRange(argmax(Pxx(FMask,NumPC),1))*60;