clc
clear all
close all

%Best Configuration => (Delay 20, Node 15)
%Runner Up Configuration => (Delay 20, Node 10)
%3rd place Configuration => (Delay 20, Node 5)
addpath(fullfile('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/PersonX/')); 
load('Person_2.mat');

Train = [Round1Circle; Round1Triangle; Round1Right; Round1Down]; %1200x2
Train = Train'; %2x1200
Train = con2seq(Train);

Test = [Round2Circle; Round2Triangle; Round2Right; Round2Down]; %1200x2
Test = Test'; %2x1200
Test = con2seq(Test);

%Target Values
%Circle targets
CTargets = [ones(1,300); -ones(1,300); -ones(1,300); -ones(1,300)];
%Triangle targets
TTargets = [-ones(1,300); ones(1,300); -ones(1,300); -ones(1,300)];
%Right targets
RTargets = [-ones(1,300); -ones(1,300); ones(1,300); -ones(1,300)];
%Down targets
DTargets = [-ones(1,300); -ones(1,300); -ones(1,300); ones(1,300)];

Targets = [CTargets TTargets RTargets DTargets]; 
Targets = [Targets];
Targets = con2seq(Targets);

I = Train;

T = Targets;

Htotal = [];
%////////////////////////////end data///////////////////////////////////

%Create the time delay network
inputDelay = 1:30; %10 20 30
hiddenSize = 15; %5 10 15

for inputDelaysize = 10:10:30
inputDelay = 1:inputDelaysize; %10 20 30
for hiddenNodeSize = 5:5:15
hiddenSize = hiddenNodeSize; %5 10 15
net = timedelaynet(inputDelay,hiddenSize); %create network
net.layers{2}.transferFcn='tansig';
net.divideparam.trainratio = 1; %training = 100% of data
net.divideparam.valratio = 0;
net.divideparam.testratio = 0; 
net.trainFcn='trainbr'; % use Bayesian regularization backpropagation
net.trainParam.epochs = 100; % 100 epochs
[Xs,Xi,Ai,Ts] = preparets(net,I,T);

for k = 1:5
net = init(net);%initialize network
[net,tr] = train(net,Xs, Ts); %train network

%test network
testNet = net(Test, Xi, Ai); %??????????????????????????????????wut

testNet = cell2mat(testNet);

%Calculate Classifier output matrices

H = [];


for i = 1:4
    for j = 1:4
        H(j,i) = sum(testNet(i,300*(j-1)+1:300*j))/(size(testNet,2)/4); %size of testnet/4 = 300
    end
end
Htotal = cat(3,Htotal,H);
end
%ezroc3(cat(3,H1,H2,H3));
ezroc3(Htotal);
%savefig
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/BestConfigROCFigureX/FigureDelay_' num2str(inputDelaysize) '_Node_' num2str(hiddenNodeSize) '.fig']);

end
end




