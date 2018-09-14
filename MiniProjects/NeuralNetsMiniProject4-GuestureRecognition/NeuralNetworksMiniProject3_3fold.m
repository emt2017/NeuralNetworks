clc
clear all
close all

Htotal = []; %Final H value used for ROC curve
Hmatrix = []; % H value that will hold H1,H2,H3....H(crossfold)
Havg = []; %average of all values in Hmatrix (used for Extra Credit ROC)
crossfold = 3; %define crossfold
inputDelay = 1:25;%use best delay
outputDelay = 1:10;%use best output
hiddenSize = 10;%use best node size
BestInput = 25;
BestOutput = 10;
BestSize = 10;
personNum = 1;
crossNum = 1;

for personNum = 1:6
    addpath(fullfile('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/PersonX/')); 
    personFileName = sprintf('Person_%d.mat',personNum)
    load(personFileName);
    Htotal = []; %reset Htotal
%for inputdelay
%for hiddenSize
%calc Havg
%Hmatrix <= Havg    
for crossNum = 1:crossfold
%Crossfold Conditions
if crossNum == 1 
Train = [Round1Circle; Round1Triangle; Round1Right; Round1Down; Round2Circle; Round2Triangle; Round2Right; Round2Down]; %2400x2
Test = [Round3Circle; Round3Triangle; Round3Right; Round3Down]; %1200x2
end
if crossNum == 2
Train = [Round1Circle; Round1Triangle; Round1Right; Round1Down; Round3Circle; Round3Triangle; Round3Right; Round3Down]; %2400x2
Test = [Round2Circle; Round2Triangle; Round2Right; Round2Down]; %1200x2
end
if crossNum == 3
Train = [Round2Circle; Round2Triangle; Round2Right; Round2Down; Round3Circle; Round3Triangle; Round3Right; Round3Down]; %2400x2
Test = [Round1Circle; Round1Triangle; Round1Right; Round1Down]; %1200x2
end

%Invert and convert data
Train = Train'; %2x2400
Train = con2seq(Train);
Test = Test'; %2x1200
Test = con2seq(Test);

%Create Target Values
%Circle targets
CTargets = [ones(1,300); -ones(1,300); -ones(1,300); -ones(1,300)];
%Triangle targets
TTargets = [-ones(1,300); ones(1,300); -ones(1,300); -ones(1,300)];
%Right targets
RTargets = [-ones(1,300); -ones(1,300); ones(1,300); -ones(1,300)];
%Down targets
DTargets = [-ones(1,300); -ones(1,300); -ones(1,300); ones(1,300)];

Targets = [CTargets TTargets RTargets DTargets]; 
Targets = [Targets Targets];
Targets = con2seq(Targets);

I = Train;

T = Targets;

%////////////////////////////end data///////////////////////////////////

net = narxnet(inputDelay,outputDelay,hiddenSize); %create network
net.layers{2}.transferFcn='tansig';
net.divideparam.trainratio = 1; %training = 100% of data
net.divideparam.valratio = 0;
net.divideparam.testratio = 0; 
net.trainFcn='trainbr'; % use Bayesian regularization backpropagation
net.trainParam.epochs = 100; % 100 epochs
[Xs,Xi,Ai,Ts] = preparets(net,I,{},T);

[net,tr] = train(net,Xs,Ts,Xi,Ai); %train network

%close the loop before testing
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
[Xc,Xic,Aic,Tc] = preparets(netc,I,{},T);
y2 = netc(Xc,Xic,Aic);
closedLoopPerformance = perform(netc,Tc,y2);

%test network
testNet = netc(Test, Xic, Aic); %??????????????????????????????????wut

testNet = cell2mat(testNet); %convert test output from cell to matrix

%Calculate Classifier output matrices
H = [];
%4x1200
for i = 1:4
    for j = 1:4
        H(j,i) = sum(testNet(i,300*(j-1)+1:300*j))/(size(testNet,2)/4);
    end
end

Htotal = cat(3,Htotal,H);

%save net??????????????????????????????????????????????????????????????????
%save (['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject4/SubjectSpecificNets/NetDelay_' num2str(BestInput) '_Node_' num2str(BestSize) '_Person_' num2str(personNum) '_fold_' num2str(crossNum) '.mat'], 'net');


end

%ROC Curve
%save(Htotal);

ezroc3(Htotal);
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/3FoldROCCurves/FigureDelay_' num2str(BestDelay) '_Node_' num2str(BestSize) '_Person_' num2str(personNum) '.fig']);
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject4/3FoldROCCurves/Person_' num2str(personNum) 'FigureInput_' num2str(BestInput) '_Output_' num2str(BestOutput) '_Node_' num2str(BestSize) '.fig']);
end

%ezroc3(Havg);
%networks = 6 people x 3 folds = 18 saved networks
%netwroks = 6 folds = 6 saved networks
