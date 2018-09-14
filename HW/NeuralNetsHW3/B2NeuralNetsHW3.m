close all;
clear all;
clc;
%/////////////////////////////////////FULL DATA////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1:9,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(1:9,350:699)'*w; %Activation/testing (we'll talk about big no no here later)


%ezroc3(X(3,:),d(2,:));
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC FULL DATA, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 1///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(1,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 1, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 2///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(2,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(2,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 2, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 3///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(3,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(3,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 3, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 4///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(4,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(4,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 4, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 5///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(5,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(5,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 5, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 6///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(6,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(6,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 6, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 7///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(7,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(7,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 7, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 8///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(8,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(8,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 8, AUC=' num2str(AUC)])

%//////////////////////////////////SUBSET 9///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(9,1:349)'\d(2,1:349)'; %Training/MSE linear model creation
y=X(9,350:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,350:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 9, AUC=' num2str(AUC)])