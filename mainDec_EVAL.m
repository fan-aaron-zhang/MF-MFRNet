function mainDec_MFRnet_HDR(scriptPath,hostDecoder,streamPath,yuvPath,origPath,statPath,no)

%% for debug;
% for debug only;
% scriptPath='D:/work/prog/ViSTRA3/DEC/';
% hostDecoder='HM1620_Decoder_WIN7.exe';
% streamPath='E:\work\data\ViSTRA\HM1620_JVET_SR\STREAM\';
% yuvPath='E:\work\data\ViSTRA\HM1620_JVET_SR\DEC\';
% origPath='M:\Aaron\ViSTRA\JVET_CTC\SDR_10bit\';
% statPath='E:\work\data\ViSTRA\HM1620_JVET_SR\STAT\';
%

%%%%%%%%%%%%%%%%%%%%%%%%%%

hostDecoder = [scriptPath 'bin/' hostDecoder];
[~,tmpName,~] = fileparts(hostDecoder);
idx = strfind(tmpName,'_');
VERSION = tmpName(1:idx(1)-1);

[seqName1,streamFiles,width,height,~,~,bitDepth_rec,~,QP,SR_flag,BD_flag] = getOrigFiles1(streamPath,['.' lower(VERSION)]);
[seqName,origFile,full_width,full_height,noOfFrames,fps,bitDepth] = getOrigFiles(origPath,'.yuv');
[full_width full_height noOfFrames fps bitDepth]

for s = 1:size(seqName(:),1)
    if (strcmp(seqName{s},seqName1{no}) == 1)
        src = s;
        break;
    end
end

filelist =  dir(fullfile(streamPath,['*.' lower(VERSION)]));
if (size(filelist(:),1) == 0)
    error ('There are no video stream files at the given location!');
elseif (size(filelist(:),1) < no)
    error ('The stream file has not been found!');
end

if (exist(yuvPath,'dir') ==0)
    mkdir(yuvPath);
end

streamFile = fullfile(streamPath,[filelist(no).name]);
recFile = fullfile(yuvPath,[streamFiles{no} '.yuv']);

if (exist(recFile,'file')~=0)
    d = dir(recFile);
    tmp_no = d.bytes/ceil(bitDepth(src)/8)/height(no)/width(no)/1.5;
    
    if (tmp_no~=noOfFrames(src))
        fprintf ('The video is still being encoded!');
        return;
    end
    
else
    if (exist(yuvPath,'dir') ==0)
        mkdir(yuvPath);
    end
    
    system(sprintf... decoder parameters;
        (['"%s" ' ...binfile full name;
        '-b "%s" ' ... binary bitstream file;
        '-o "%s"'], ... decoded yuv file;
        ...
        hostDecoder,...
        streamFile,...
        recFile));
end

%% Upsampling

idx = strfind(streamFiles{no},'_');
    recFile_tmp        = fullfile(yuvPath ,[seqName1{no}  '_' num2str(full_width(src)) 'x' num2str(full_height(src))  streamFiles{no}(idx(2):end) '_tmp.yuv']);
    
if (SR_flag(no)~=0)
    
    d = dir(recFile);
    rec_noOfFrames= d.bytes/ceil(bitDepth(src)/8)/width(no)/height(no)/1.5;
    
    for f = 1:rec_noOfFrames
        origYuv = loadFileYuv(recFile, width(no),height(no),bitDepth(src),f);
        testYuv = origYuv(1:full_height(src)/2^SR_flag(no),1:full_width(src)/2^SR_flag(no),:);
        tmpYuv(:,:,1) = imresize(testYuv(:,:,1),2^SR_flag(no),'nearest');
        tmpYuv(:,:,2) = imresize(imresize(imresize(testYuv(:,:,2),0.5,'nearest'),2^SR_flag(no),'nearest'),2,'nearest');
        tmpYuv(:,:,3) = imresize(imresize(imresize(testYuv(:,:,3),0.5,'nearest'),2^SR_flag(no),'nearest'),2,'nearest');
        
        saveFileYuv(uint16(tmpYuv), recFile_tmp, 10, f)
    end
    
    delete(recFile);
    recFile = recFile_tmp;
else
    movefile(recFile,recFile_tmp);
    recFile = recFile_tmp;
end

idx = strfind(streamFiles{no},'_');
decFile             = fullfile(yuvPath ,[origFile{src} streamFiles{no}(idx(5):end) '.yuv']);



if (SR_flag(no)+BD_flag(no)==0)
    if (strcmp(decFile,recFile)==0)
        movefile(recFile,decFile);
        recFile = decFile;
    end
    %% Using CNN for BD upsampling;
else
    hostCNN = [scriptPath 'CNN/'];
    modelPath = [hostCNN 'models/' 'VTM162_BD_MFRNet_ORI/'];
    %modelPath = ['/mnt/storage/scratch/cf18202/CNN_Di/Z_Evaluation_Models/' 'BD_VTM70_HDR/'];
    
    modelFile = [modelPath 'MFRNet_VTM162'];
    if (SR_flag(no)>0)
        modelFile = [modelFile 'SRGAN'];
        QPGroup = QP(no) + 6;
    end
    
    if (BD_flag(no)>0)
        modelFile = [modelFile '_BD'];
        %QPGroup = QPGroup + 6;
        QPGroup = QP(no) + 6;
		%QPGroup = QP(no);
    end
    
    modelFile = [modelFile '_QP' num2str(QPGroup) '.npz'];
	
	modelFile
	
    system(sprintf(['python %s' 'main_GAN_VGG_PSNR_MFRNet.py --mode=evaluate ' ...
        '--evalHR=%s '...
        '--evalLR=%s '...
        '--testModel=%s '...
        '--ratio=1 '...
        '--nlayers=16 '...
        '--GAN=0 '...
        '--nframes=0 '...
        '--eval_inputType=YUV '...
        '--readBatch_flag=1 '...
        '--inputFormat=RGB '],...
        ...
        hostCNN,...
        decFile,...
        recFile,...
        modelFile...
        ));
end

delete(recFile);

origFile        = [origPath origFile{src} '.yuv'];
statFile = [statPath streamFiles{no} '.mat'];

if (exist(statFile,'file')~=0)
    fprintf('Stat file has been created!\n');
    return;
end


if (exist(statPath,'dir') ==0)
    mkdir(statPath);
end


d = dir(streamFile);
bitRate = d.bytes*8/noOfFrames(src)*fps(src);

fprintf('original sequence: %s\n', origFile);
fprintf('distorted sequence: %s\n', decFile);

d = dir(decFile);
dec_noOfFrames= d.bytes/ceil(bitDepth(src)/8)/full_width(src)/full_height(src)/1.5;

if (dec_noOfFrames ~= noOfFrames(src))
    fprintf('The reconstructed file has not been fully decoded! %d and %d \n',dec_noOfFrames,noOfFrames(src));
    return;
end



for f = 1:noOfFrames(src)
    origYuv = loadFileYuv(origFile, full_width(src),full_height(src),bitDepth(src),f);
    
    if (bitDepth(src)==8)
        origYuv = double(origYuv) * 4;
    end
    
    testYuv = loadFileYuv(decFile,full_width(src),full_height(src),10,f);
    
    [PSNR_FRM(1,f),MSE_FRM(1,f)] = psnr_bitDepth(origYuv(:,:,1),testYuv(:,:,1),10);
    [PSNR_FRM(2,f),MSE_FRM(2,f)] = psnr_bitDepth(origYuv(:,:,2),testYuv(:,:,2),10);
    [PSNR_FRM(3,f),MSE_FRM(3,f)] = psnr_bitDepth(origYuv(:,:,3),testYuv(:,:,3),10);
    fprintf('Calculating PSNR for Frame No. %d/%d\n', f, noOfFrames(src));
end

c = seqName1{no}(2);
if (c == '1')
    for f = 1:noOfFrames(src)
        origYuv = loadFileYuv(origFile, full_width(src),full_height(src),bitDepth(src),f);
        
        if (bitDepth(src)==8)
            origYuv = double(origYuv) * 4;
        end
        
        testYuv = loadFileYuv(decFile,full_width(src),full_height(src),10,f);
        
        [wPSNR_FRM(1,f),wMSE_FRM(1,f)] = wpsnr_bitDepth_H1(origYuv(:,:,1),testYuv(:,:,1),10);
        [wPSNR_FRM(2,f),wMSE_FRM(2,f)] = wpsnr_bitDepth_H1(origYuv(:,:,2),testYuv(:,:,2),10);
        [wPSNR_FRM(3,f),wMSE_FRM(3,f)] = wpsnr_bitDepth_H1(origYuv(:,:,3),testYuv(:,:,3),10);
        
        fprintf('Calculating wPSNR for Frame No. %d/%d\n', f, noOfFrames(src));
    end
else if (c == '2')
        for f = 1:noOfFrames(src)
            origYuv = loadFileYuv(origFile, full_width(src),full_height(src),bitDepth(src),f);
            
            if (bitDepth(src)==8)
                origYuv = double(origYuv) * 4;
            end
            
            testYuv = loadFileYuv(decFile,full_width(src),full_height(src),10,f);
            
            [wPSNR_FRM(1,f),wMSE_FRM(1,f)] = wpsnr_bitDepth_H2(origYuv(:,:,1),testYuv(:,:,1),10);
            [wPSNR_FRM(2,f),wMSE_FRM(2,f)] = wpsnr_bitDepth_H2(origYuv(:,:,2),testYuv(:,:,2),10);
            [wPSNR_FRM(3,f),wMSE_FRM(3,f)] = wpsnr_bitDepth_H2(origYuv(:,:,3),testYuv(:,:,3),10);
            
            fprintf('Calculating wPSNR for Frame No. %d/%d\n', f, noOfFrames(src));
        end
    end
end

PSNR_SQ = 20*log10((2^(10)-1)/(sqrt(mean(MSE_FRM(1,:)))));
% save(statFile,'PSNR_FRM','PSNR_SQ','bitRate','MSE_FRM');
wPSNR_SQ = 20*log10((2^(10)-1)/(sqrt(mean(wMSE_FRM(1,:)))));
save(statFile,'wPSNR_FRM','wPSNR_SQ','bitRate','wMSE_FRM','PSNR_FRM','PSNR_SQ','MSE_FRM');

%delete(decFile);


function [decibels,mse] = psnr_bitDepth(A,B,BD)

error_diff = double(A) - double(B);
mse = mean(error_diff(:).^2);
decibels = 20*log10((2^(BD)-1)/(sqrt(mse)));

function [decibels,wmse] = wpsnr_bitDepth_H1(A,B,BD)

error_diff = double(A) - double(B);
y = 0.03.*double (A) - 3.0 ;
y(y>12) = 12;
y(y<0) = 0;
w = 2.0.^(y./3.0);
mse = w.*(error_diff.^2);
wmse = mean(mse(:));  
% mse = mean(error_diff(:).^2);
% decibels = 20*log10((2^(BD)-1)/(sqrt(mse)));
decibels = 20*log10((2^(BD)-1)/(sqrt(wmse)));


function [decibels,wmse] = wpsnr_bitDepth_H2(A,B,BD)

error_diff = double(A) - double(B);
y = 0.015.*double (A) - 1.5 - 6 ;
y(y>6) = 6;
y(y<-3) = -3;
w = 2.0.^(y./3.0);
mse = w.*(error_diff.^2);
wmse = mean(mse(:));  
% mse = mean(error_diff(:).^2);
% decibels = 20*log10((2^(BD)-1)/(sqrt(mse)));
decibels = 20*log10((2^(BD)-1)/(sqrt(wmse)));


%% Read coding parameters from original test sequences;
function [seqName,inputFile,width,height,noOfFrames,fps,bitDepth,intraPeriod,SR_flag,BD_flag] = getOrigFiles(testPath,Extention)


filelist = dir([testPath '*' Extention]);
seqNo = 0;

for s = 1:size(filelist(:),1)
    seqNo = seqNo +1;
    
    idx = strfind(filelist(s).name,Extention);
    inputFile{seqNo} = filelist(s).name(1:idx(1)-1);
    
    idx1 = strfind(filelist(s).name,'_');
    seqName{seqNo} = filelist(s).name(1:idx1(1)-1);
    
    idx2 = strfind(filelist(s).name,'x');
    idx2 = idx2(idx2>idx1(1));
    width(seqNo) = str2double(filelist(s).name(idx1(1)+1:idx2(1)-1));
    height(seqNo) = str2double(filelist(s).name(idx2(1)+1:idx1(2)-1));
    fps(seqNo) = str2double(filelist(s).name(idx1(2)+1:idx1(3)-4));
    
    idx3 = strfind(filelist(s).name(1:end),'bit');
    bitDepth(seqNo) = str2double(filelist(s).name(idx1(3)+1:idx3(1)-1));
    
    d = dir([testPath filelist(s).name]);
    noOfFrames(seqNo) = d.bytes/ceil(bitDepth(seqNo)/8)/height(seqNo)/width(seqNo)/1.5;
    
end

seqName = seqName(:);
inputFile = inputFile(:);
width  = width(:);
height = height(:);
noOfFrames = noOfFrames(:);
fps = fps(:);
bitDepth = bitDepth(:);
intraPeriod = double(int16(fps/16)*16);


function [seqName,inputFile,width,height,noOfFrames,fps,bitDepth,intraPeriod,QP,SR_flag,BD_flag] = getOrigFiles1(testPath,Extention)

filelist = dir([testPath '*' Extention]);
seqNo = 0;

for s = 1:size(filelist(:),1)
    seqNo = seqNo +1;
    
    idx = strfind(filelist(s).name,Extention);
    inputFile{seqNo} = filelist(s).name(1:idx(1)-1);
    
    idx1 = strfind(filelist(s).name,'_');
    seqName{seqNo} = filelist(s).name(1:idx1(1)-1);
    
    idx2 = strfind(filelist(s).name,'x');
    idx2 = idx2(idx2>idx1(1));
    width(seqNo) = str2double(filelist(s).name(idx1(1)+1:idx2(1)-1));
    height(seqNo) = str2double(filelist(s).name(idx2(1)+1:idx1(2)-1));
    fps(seqNo) = str2double(filelist(s).name(idx1(2)+1:idx1(2)+2));
    
    idx3 = strfind(filelist(s).name(1:end),'BD');
    
    if (size(idx3,1) == 0)
        bitDepth(seqNo) = 10;
    else
        bitDepth(seqNo) = str2double(filelist(s).name(idx3(end)+2:idx(1)-1))/10;
    end
    
    idx4 = strfind(filelist(s).name(1:end),'_BD9');
    
    if (size(idx4,1) == 0)
        BD_flag(seqNo) = 0;
    else
        BD_flag(seqNo) = 1;
    end
    
    idx5 = strfind(filelist(s).name(1:end),'_SR20');
    
    if (size(idx5,1) == 0)
        SR_flag(seqNo) = 0;
    else
        SR_flag(seqNo) = 1;
    end
    
    
    idx4 = strfind(filelist(s).name(1:end),'_qp');
    QP(seqNo) = str2double(filelist(s).name(idx4(1)+3:idx4(1)+4));
    
    d = dir([testPath filelist(s).name]);
    noOfFrames(seqNo) = d.bytes/ceil(bitDepth(seqNo)/8)/height(seqNo)/width(seqNo)/1.5;
    
end

seqName = seqName(:);
inputFile = inputFile(:);
width  = width(:);
height = height(:);
noOfFrames = noOfFrames(:);
fps = fps(:);
bitDepth = bitDepth(:);
intraPeriod = double(int16(fps/16)*16);
QP = QP(:);
%% Save YUV files;
function saveFileYuv(imgYuv, fileName, bitDepth, mode)

precision = ceil(double(bitDepth)/8);

if (precision==1)
    strPrecision = 'uint8';
    imgYuv = uint8(imgYuv);
elseif (precision==2)
    strPrecision = 'uint16';
    imgYuv = uint16(imgYuv);
else
    error('Error: Input bit depth is not correct!\n');
end

switch mode
    case 1 % replace file
        fileId = fopen(fileName, 'w');
    otherwise
        fileId = fopen(fileName, 'a');
end

imgYuv = double(imgYuv);
% write Y component
buf = reshape(imgYuv(:, :, 1).', [], 1); % reshape
fwrite(fileId, buf, strPrecision);

% write U component
buf = reshape(imgYuv(1 : 2 : end, 1 : 2 : end, 2).', [], 1); % downsample and reshape
fwrite(fileId, buf, strPrecision);

% write V component
buf = reshape(imgYuv(1 : 2 : end, 1 : 2 : end, 3).', [], 1); % downsample and reshape
fwrite(fileId, buf, strPrecision);

fclose(fileId);

%% Load YUV files;
function [imgYuv] = loadFileYuv(fileName, width, height, bitDepth,idxFrame)
% load RGB movie [0, 255] from YUV 4:2:0 file

precision = ceil(double(bitDepth)/8);

if (precision==1)
    strPrecision = 'uint8';
elseif (precision==2)
    strPrecision = 'uint16';
else
    error('Error: Input bit depth is not correct!\n');
end

fileId = fopen(fileName, 'r');

subSampleMat = [1, 1; 1, 1];
nrFrame = length(idxFrame);

for f = 1 : 1 : nrFrame
    % search fileId position
    sizeFrame = 1.5 * width * height * precision;
    fseek(fileId, (idxFrame(f) - 1) * sizeFrame, 'bof');
    
    % read Y component
    buf = fread(fileId, width * height, strPrecision);
    imgYuv(:, :, 1) = reshape(buf, width, height).'; % reshape
    
    % read U component
    buf = fread(fileId, width / 2 * height / 2, strPrecision);
    imgYuv(:, :, 2) = kron(reshape(buf, width / 2, height / 2).', subSampleMat); % reshape and upsample
    
    % read V component
    buf = fread(fileId, width / 2 * height / 2, strPrecision);
    imgYuv(:, :, 3) = kron(reshape(buf, width / 2, height / 2).', subSampleMat); % reshape and upsample
    
    % normalize YUV values
    if (precision==1)
        imgYuv = uint8( imgYuv );
    elseif (precision==2)
        imgYuv = uint16( imgYuv );
    end
    
end
fclose(fileId);
