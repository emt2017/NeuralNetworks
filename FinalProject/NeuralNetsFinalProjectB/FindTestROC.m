clc
clear all
close all

%Load the training data into memory
%load FaceTestAndTrainCellArrayData.mat
%load(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectB\PersonTrain\Person' num2str(i) 'Train.mat']);

% img = imread('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA\Face_Data\s1\1.pgm')
%  img = FaceTrainCellArrayData{1}
%  imshow(img);
%even numbers on all except for Sparsity Proportion

tTestF = [];
yF = [];

for i = 1:40

load(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB\PersonTest\Person' num2str(i) 'Test.mat']);
    
%normalize data
FaceTestCellArrayData;

% Get the number of pixels in each image
imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(FaceTestCellArrayData));
for i = 1:numel(FaceTestCellArrayData)
    xTest(:,i) = FaceTestCellArrayData{i}(:);
end

xTest = xTest/255;

TestCell=[];
for i=1:4
  Test=reshape(xTest(:,i),112,92);
  TestCell=[TestCell {Test}];
end

FaceTestCellArrayData=TestCell;


% Get the number of pixels in each image
imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(FaceTestCellArrayData));
for i = 1:numel(FaceTestCellArrayData)
    xTest(:,i) = FaceTestCellArrayData{i}(:);
end

for j=1:40
    %load network based on j
    load(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB\nets\Person' num2str(j) '.mat']);
    y = deepnet(xTest);
    yF = cat(2,yF,y);
end

tTestEz = [ones(1,4) zeros(1,156)];

tTestF = cat(2,tTestF,tTestEz);

end
%save yF and tTrainF
%y = deepnet(feat2);    

ezroc3(yF,tTestF); %saveezroc3?
%ezroc3(-1*[ezrocG ezrocI],[ezrocGTargets ezrocITargets]);