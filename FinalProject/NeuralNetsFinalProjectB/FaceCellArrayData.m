clc
clear all
close all

%Train
count = 7;
FaceTrainCellArrayData = [];

for i = 1:40    
    count = 7;
    FaceTrainCellArrayData = [];
    for j = 1:6
        img = imread(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTrainCellArrayData{j}=img;
    end
    
    for k = 1:40
        for j = 1:6
            if k~=i
                img = imread(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB\Face_Data\s' num2str(k) '\' num2str(j) '.pgm']);
                FaceTrainCellArrayData{count}=img;
                count = count + 1;
            end
        end
    end
    save(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB/PersonTrain/Person' num2str(i) 'Train.mat'], 'FaceTrainCellArrayData');

end

%Test 
count = 1;
FaceTestCellArrayData = cell(1, 4);

for i = 1:40    
    
    count = 1;
    FaceTestCellArrayData = cell(1, 4);
    
    for j = 7:10
        img = imread(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTestCellArrayData{count}=img;
        count = count + 1;
    end
    
%     for k = 1:40
%         for j = 7:10
%             if k~=i
%                 img = imread(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB\Face_Data\s' num2str(k) '\' num2str(j) '.pgm']);
%                 FaceTestCellArrayData{count}=img;
%                 count = count + 1;
%             end
%         end
%     end
    save(['C:\Users\SWAT SEAL\Documents\MATLAB\NeuralNetsFinalProjectB/PersonTest/Person' num2str(i) 'Test.mat'], 'FaceTestCellArrayData');

end

% addpath(fullfile('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsFinalProjectA\')); 
% %if you are not me you will have to use a different save path
% save ('FaceCellArrayData');