%For EER evaluation 
clear; close all; clc;

addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));

outputscores1=csvread('finfeaturesGDIIeval.csv');
outputscores=outputscores1(:,1);

truefin1=zeros(1298,1);
truefin2=ones(12008,1);
truefin=[truefin1;truefin2];

% compute performance
[Pmiss,Pfa] = rocch(outputscores(truefin==0),outputscores(truefin==1));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

