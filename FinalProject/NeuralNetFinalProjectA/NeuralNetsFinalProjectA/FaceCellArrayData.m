clc
clear all
close all

%Train
i = 1; %person 1-40
j = 1; %img 1-6
count = 0;
FaceTrainCellArrayData = cell(1, count);

for i = 1:40    
    for j = 1:6
        count = count + 1;  
        img = imread(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTrainCellArrayData{count}=img;
    end
end

%Test 
i = 1; %person 1-40
j = 1; %img 7-10
count = 0;
FaceTestCellArrayData = cell(1, count);

for i = 1:40    
    for j = 7:10
        count = count + 1;  
        img = imread(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTestCellArrayData{count}=img;
    end
end

% addpath(fullfile('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA\')); 
% %if you are not me you will have to use a different save path
% save ('FaceCellArrayData');