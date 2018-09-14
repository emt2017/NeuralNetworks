clc
clear all
close all

%load FaceTestAndTrainCellArrayData.mat
load Person1Train.mat
%img = imread('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA\Face_Data\s1\1.pgm')
%imshow(img);
%even numbers on all except for Sparsity Proportion

hiddenSize = 100;
L2Weight = 0.004;
SparsityReg = 2;
SparsityProportion = 0.15;%no less than 0.05
% 
% for hiddenSize = 200:100:200
%     for L2Weight = 0.001:0.015:0.016
%         for SparsityReg = 8:6:8
%             for SparsityProportion = 0.15:0.15:0.3

% Load the training data into memory

% for hiddenSize = 200:100:200
%     for L2Weight = 0.001:0.015:0.016
%         for SparsityReg = 8:6:8
%             for SparsityProportion = 0.15:0.15:0.3

%Load the training data into memory

%normalize data
FaceTrainCellArrayData;
FaceTestCellArrayData;

% Get the number of pixels in each image
imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;

% Turn the test images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(FaceTrainCellArrayData));
for i = 1:numel(FaceTrainCellArrayData)
    xTrain(:,i) = FaceTrainCellArrayData{i}(:);
end

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(FaceTestCellArrayData));
for i = 1:numel(FaceTestCellArrayData)
    xTest(:,i) = FaceTestCellArrayData{i}(:);
end

xTrain = xTrain/255;
xTest = xTest/255;

TrainCell=[];
for i=1:240
  Train=reshape(xTrain(:,i),112,92);
  TrainCell=[TrainCell {Train}];
end

TestCell=[];
for i=1:160
  Test=reshape(xTest(:,i),112,92);
  TestCell=[TestCell {Test}];
end

FaceTrainCellArrayData=TrainCell;

FaceTestCellArrayData=TestCell;

% tTrain = zeros(40,240);
% count = 1;
% for j = 1:6:240
%     tTrain(count,j:j+5)=ones(1,6);
%     count = count + 1;
% end



rng('default')
hiddenSize1 = hiddenSize;
autoenc1 = trainAutoencoder(FaceTrainCellArrayData,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',L2Weight, ...
    'SparsityRegularization',SparsityReg, ...
    'SparsityProportion',SparsityProportion, ...
    'ScaleData', false);

plotWeights(autoenc1);
feat1 = encode(autoenc1,FaceTrainCellArrayData);
hiddenSize2 = hiddenSize/2;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',L2Weight/2, ...
    'SparsityRegularization',SparsityReg, ...
    'SparsityProportion',SparsityProportion-0.05, ...
    'ScaleData', false);
plotWeights(autoenc2);
feat2 = encode(autoenc2,feat1);

tTrain = [ones(1,6) -ones(1,234)];

net = patternnet(hiddenSize2);
net = train(net,feat2,tTrain);

%//////////////////////////////////////////////////////////
deepnet = stack(autoenc1,autoenc2,net);
%save (['\\kc.umkc.edu\kc- users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA/nets/HiddenSize_' num2str(hiddenSize) '_L2Weight_' num2str(L2Weight) '_SparsityReg_' num2str(SparsityReg) '_SparsityProportion_' num2str(SparsityProportion) '.mat'], 'deepnet');

% Get the number of pixels in each image
imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;

% Turn the test images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(FaceTrainCellArrayData));
for i = 1:numel(FaceTrainCellArrayData)
    xTrain(:,i) = FaceTrainCellArrayData{i}(:);
end
y = deepnet(xTrain);

%//////////Find MSE of guenuine & imposter features////////////
count = 1;
ezrocG = zeros(1, 600); % make sure to preallocate my matrices
ezrocI = zeros(1, 28080);
ezrocGTargets = ones(1, 600); % make sure to preallocate my matrices
ezrocITargets = zeros(1, 28080);
for person = 1:40
    for img = 1:6
        for i = img+1:6
            %where img is current img and img+1 is the next image
            ezrocG(count) = immse(feat2(1:hiddenSize/2,(img+(person-1)*6)),feat2(1:hiddenSize/2,(i+(person-1)*6)));
            count=count+1;
        end
    end
end

count = 1;
for person = 1:40
    for img = 1:6
        for personComp = person + 1:40
        %compare personComp +1 then iterate 39times
            for imgComp = 1:6
                ezrocI(count) = immse(feat2(1:hiddenSize/2,(img+(person-1)*6)),feat2(1:hiddenSize/2,(imgComp+(personComp-1)*6)));
                count=count+1;
            end
        end
    end
end

ezroc3(y,tTrain);
ezroc3(-1*[ezrocG ezrocI],[ezrocGTargets ezrocITargets]);
%save in file ezrocTrain hiddenSize,L2Weight,SparsityReg,SparsityProportion
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA/ezrocTrain/FigureHiddenSize_' num2str(hiddenSize) '_L2Weight_' num2str(L2Weight) '_SparsityReg_' num2str(SparsityReg) '_SparsityProportion_' num2str(SparsityProportion) '.fig']);
%save trained network

% Get the number of pixels in each image
imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(FaceTestCellArrayData));
for i = 1:numel(FaceTestCellArrayData)
    xTest(:,i) = FaceTestCellArrayData{i}(:);
end
y = deepnet(xTest);

%//////////Find MSE of guenuine & imposter features Test////////////Error
count = 1;
ezrocG = zeros(1, 240); % make sure to preallocate my matrices
ezrocI = zeros(1, 12480);
ezrocGTargets = ones(1, 240); 
ezrocITargets = zeros(1, 12480);
for person = 1:40
    for img = 1:4
        for i = img+1:4
            %where img is current img and img+1 is the next image
            ezrocG(count) = immse(y(1:hiddenSize/2,(img+(person-1)*4)),y(1:hiddenSize/2,(i+(person-1)*4)));
            count=count+1;
        end
    end
end

count = 1;
for person = 1:40
    for img = 1:4
        for personComp = person + 1:40
        %compare personComp +1 then iterate 39times
            for imgComp = 1:4
                ezrocI(count) = immse(y(1:hiddenSize/2,(img+(person-1)*4)),y(1:hiddenSize/2,(imgComp+(personComp-1)*4)));
                count=count+1;
            end
        end
    end
end

ezroc3(-1*[ezrocG ezrocI],[ezrocGTargets ezrocITargets]);
%save in file ezrocTest hiddenSize,L2Weight,SparsityReg,SparsityProportion
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA/ezrocTest/FigureHiddenSize_' num2str(hiddenSize) '_L2Weight_' num2str(L2Weight) '_SparsityReg_' num2str(SparsityReg) '_SparsityProportion_' num2str(SparsityProportion) '.fig']);

%             end
%         end
%     end
% end
