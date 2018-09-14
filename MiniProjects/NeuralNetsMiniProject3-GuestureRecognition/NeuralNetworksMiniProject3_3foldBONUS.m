clc
clear all
close all

Htotal = []; %Final H value used for ROC curve
Hmatrix = []; % H value that will hold H1,H2,H3....H(crossfold)
Havg = []; %average of all values in Hmatrix (used for Extra Credit ROC)
Hsum = zeros(4,4);
crossfold = 3; %define crossfold
inputDelay = 1:20;%use best delay
hiddenSize = 15;%use best node size
BestDelay = 20;
BestSize = 15;
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

for committee = 1:2
if committee == 1
inputDelay = 1:20;%use best delay
hiddenSize = 15;%use best node size
BestDelay = 20;
BestSize = 15; 
end
if committee == 2
inputDelay = 1:20;%use best delay
hiddenSize = 10;%use best node size
BestDelay = 20;
BestSize = 10;     
end
% if committee == 3
% inputDelay = 1:20;%use best delay
% hiddenSize = 5;%use best node size
% BestDelay = 20;
% BestSize = 5;    
% end
  load(personFileName);
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

%Create the time delay network
net = timedelaynet(inputDelay,hiddenSize); %create network
net.layers{2}.transferFcn='tansig'; %change to tansig function
net.divideparam.trainratio = 1; %training = 100% of data 
net.divideparam.valratio = 0; %disable early stopping
net.divideparam.testratio = 0; 
net.trainFcn='trainbr'; % use Bayesian regularization backpropagation
net.trainParam.epochs = 100; % 100 epochs
[Xs,Xi,Ai,Ts] = preparets(net,I,T);

net = init(net);%initialize network
[net,tr] = train(net,Xs, Ts); %train network
%save (['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/SubjectSpecificNetsBONUS/NetDelay_' num2str(BestDelay) '_Node_' num2str(BestSize) '_Person_' num2str(personNum) '_fold_' num2str(crossNum) '.mat'], 'net');
%test network
testNet = net(Test, Xi, Ai);

testNet = cell2mat(testNet); %convert test output from cell to matrix

%Calculate Classifier output matrices
H = [];
%4x1200
for i = 1:4
    for j = 1:4
        H(j,i) = sum(testNet(i,300*(j-1)+1:300*j))/(size(testNet,2)/4);
    end
end
    Hsum = Hsum + H;
end
Havg = Hsum/3;
Htotal = cat(3,Htotal,Havg);

%save net??????????????????????????????????????????????????????????????????



end
%ROC Curve
%save(Htotal);

ezroc3(Htotal);
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/3FoldROCCurvesBONUS/FigureDelay_' num2str(BestDelay) '_Node_' num2str(BestSize) '_Person_' num2str(personNum) '.fig']);

end



%ezroc3(Havg);
%networks = 6 people x 3 folds = 18 saved networks
%netwroks = 6 folds = 6 saved networks