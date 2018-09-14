close all;
clear all;
clc;
%/////////////////////////////////////FULL INPUT////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X'\d(2,:)'; %Training/MSE linear model creation
y=X'*w; %Activation/testing (we'll talk about big no no here later)

%ezroc3(X(3,:),d(2,:));
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC FULL INPUT, AUC=' num2str(AUC)])

%/////////////////////////////////////SUBSET OF FULL INPUT////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1:9,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(1:9,1:350)'*w; %Activation/testing (we'll talk about big no no here later)

%ezroc3(X(3,:),d(2,:));
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET OF FULL INPUT, AUC=' num2str(AUC)])



%//////////////////////////////////INPUT 1///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(1,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 1///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(1,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 2///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(2,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(2,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 2, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 2///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(2,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(2,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 2, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 3///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(3,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(3,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 3, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 3///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(3,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(3,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 3, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 4///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(4,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(4,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 4, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 4///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(4,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(4,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 4, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 5///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(5,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(5,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 5, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 5///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(5,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(5,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 5, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 6///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(6,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(6,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 6, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 6///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(6,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(6,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 6, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 7///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(7,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(7,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 7, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 7///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(7,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(7,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 7, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 8///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(8,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(8,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 8, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 8///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(8,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(8,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 8, AUC=' num2str(AUC)])

%//////////////////////////////////INPUT 9///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(9,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(9,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 9, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 9///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(9,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(9,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 9, AUC=' num2str(AUC)])