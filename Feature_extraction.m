%code for extracting GD-gram features from utterances
clear; close all; clc;

pathToDatabaseT=strcat('data/ASVspoof2017/ASVspoof2017_train_dev/wav/train');
pathToDatabaseD=strcat('data/ASVspoof2017/ASVspoof2017_train_dev/wav/dev');
pathToDatabaseE=strcat('data/ASVspoof2017_eval/ASVspoof2017_eval');

trainProtocolFile='data/ASVspoof2017/ASVspoof2017_train_dev/protocol/ASVspoof2017_train.trn';
devProtocolFile='data/ASVspoof2017/ASVspoof2017_train_dev/protocol/ASVspoof2017_dev.trl';
evalProtocolFile='data/protocol/protocol/ASVspoof2017_eval_v2_key.trl.txt';

mkdir(strcat('GD1/train/0'));
mkdir(strcat('GD1/train/1'));

mkdir(strcat('GD1/val/0'));
mkdir(strcat('GD1/val/1'));

mkdir(strcat('GD1/eval/0'));
mkdir(strcat('GD1/eval/1'));

mkdir(strcat('GD2/train/0'));
mkdir(strcat('GD2/train/1'));

mkdir(strcat('GD2/val/0'));
mkdir(strcat('GD2/val/1'));

mkdir(strcat('GD2/eval/0'));
mkdir(strcat('GD2/eval/1'));





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));




disp('Extracting GD features for GENUINE dev data...');
genuineFeatureCell = cell(size(genuineIdx));
size_gen_dev=zeros(1,760);
parfor i=1:length(genuineIdx)
    
   filePath = fullfile(pathToDatabaseD,strcat(filelist{genuineIdx(i)}));
    [x1] = group_delay_feature(filePath) ; 

    if size(x1,2)>=290  
    x1=x1(:,1:290);
    end
    if mod(size(x1,2),290)~=0
    
    x1=transpose(x1);

    
    x1=padarray(x1,[290-mod(size(x1,1),290)],'circular','post');
   
    
    x1=transpose(x1);
    end

    x1=20*log10(abs(x1));
    
    x_test=x1;
    x_test(isinf(x1)) = 0.0;
    x1(isinf(x1)) = min(x_test(:));
    scaledx1 = (x1-min(x1(:))) ./ (max(x1(:)-min(x1(:))));

    imwrite(scaledx1,strcat('GD1/val/0/GDdevgen17_',num2str(i),'.png'));

end

disp('Done!');


size_spo_dev=zeros(1,950);
disp('Extracting GD features for SPOOF dev data...');

parfor i=1:length(spoofIdx)    
    
    filePath = fullfile(pathToDatabaseD,strcat(filelist{spoofIdx(i)}));
    [x1] = group_delay_feature(filePath) ; 


    if size(x1,2)>=290  
    x1=x1(:,1:290);
    end
    if mod(size(x1,2),290)~=0
    
    x1=transpose(x1);

    
    x1=padarray(x1,[290-mod(size(x1,1),290)],'circular','post');
   
    
    x1=transpose(x1);
    end

    x1=20*log10(abs(x1));
    
    x_test=x1;
    x_test(isinf(x1)) = 0.0;
    x1(isinf(x1)) = min(x_test(:));
    scaledx1 = (x1-min(x1(:))) ./ (max(x1(:)-min(x1(:))));

 
    imwrite(scaledx1,strcat('GD1/val/1/GDdevspo17_',num2str(i),'.png'));
    
end

disp('Done!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));



%extract features for GENUINE training data and store in cell array
disp('Extracting GD features for GENUINE train data...');
genuineFeatureCell = cell(size(genuineIdx));
size_gen_train=zeros(1,1508);
parfor i=1:length(genuineIdx)%changed parfor
   filePath = fullfile(pathToDatabaseT,strcat(filelist{genuineIdx(i)}));
    [x1] = group_delay_feature(filePath) ; 


    if size(x1,2)>=290  
    x1=x1(:,1:290);
    end
    if mod(size(x1,2),290)~=0
    
    x1=transpose(x1);

    
    x1=padarray(x1,[290-mod(size(x1,1),290)],'circular','post');
   
    
    x1=transpose(x1);
    end

    x1=20*log10(abs(x1));
    
    x_test=x1;
    x_test(isinf(x1)) = 0.0;
    x1(isinf(x1)) = min(x_test(:));
    scaledx1 = (x1-min(x1(:))) ./ (max(x1(:)-min(x1(:))));


    imwrite(scaledx1,strcat('GD1/train/0/GDtraingen17_',num2str(i),'.png'));

end

disp('Done!');


size_spo_train=zeros(1,1508);
disp('Extracting GD features for SPOOF train data...');
%spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)    
    
    filePath = fullfile(pathToDatabaseT,strcat(filelist{spoofIdx(i)}));
    [x1] = group_delay_feature(filePath) ; 


    if size(x1,2)>=290  
    x1=x1(:,1:290);
    end
    if mod(size(x1,2),290)~=0
    
    x1=transpose(x1);

    
    x1=padarray(x1,[290-mod(size(x1,1),290)],'circular','post');
   
    
    x1=transpose(x1);
    end

    x1=20*log10(abs(x1));
    
    x_test=x1;
    x_test(isinf(x1)) = 0.0;
    x1(isinf(x1)) = min(x_test(:));
    scaledx1 = (x1-min(x1(:))) ./ (max(x1(:)-min(x1(:))));

    imwrite(scaledx1,strcat('GD1/train/1/GDtrainspo17_',num2str(i),'.png'));
    
end

disp('Done!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileID = fopen(evalProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));
%extract features for GENUINE training data and store in cell array
disp('Extracting GD features for GENUINE eval data...');
genuineFeatureCell = cell(size(genuineIdx));
size_gen_eval=zeros(1,1298);
parfor i=1:length(genuineIdx)%changed parfor
   filePath = fullfile(pathToDatabaseE,strcat(filelist{genuineIdx(i)}));
    [x1] = group_delay_feature(filePath) ; 


    if size(x1,2)>=290  
    x1=x1(:,1:290);
    end
    if mod(size(x1,2),290)~=0
    
    x1=transpose(x1);

    
    x1=padarray(x1,[290-mod(size(x1,1),290)],'circular','post');
   
    
    x1=transpose(x1);
    end

    x1=20*log10(abs(x1));
    
    x_test=x1;
    x_test(isinf(x1)) = 0.0;
    x1(isinf(x1)) = min(x_test(:));
    scaledx1 = (x1-min(x1(:))) ./ (max(x1(:)-min(x1(:))));

    imwrite(scaledx1,strcat('GD1/eval/0/GDevalgen17_',num2str(i),'.png'));

    

end

disp('Done!');


size_spo_eval=zeros(1,12008);
disp('Extracting GD features for SPOOF eval data...');
%spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)    
    
   filePath = fullfile(pathToDatabaseE,strcat(filelist{spoofIdx(i)}));
    [x1] = group_delay_feature(filePath) ; 


    if size(x1,2)>=290  
    x1=x1(:,1:290);
    end
    if mod(size(x1,2),290)~=0
    
    x1=transpose(x1);

    
    x1=padarray(x1,[290-mod(size(x1,1),290)],'circular','post');
   
    
    x1=transpose(x1);
    end

    x1=20*log10(abs(x1));
    
    x_test=x1;
    x_test(isinf(x1)) = 0.0;
    x1(isinf(x1)) = min(x_test(:));
    scaledx1 = (x1-min(x1(:))) ./ (max(x1(:)-min(x1(:))));

    imwrite(scaledx1,strcat('GD1/eval/1/GDevalspo17_',num2str(i),'.png'));
    
end

disp('Done!');






