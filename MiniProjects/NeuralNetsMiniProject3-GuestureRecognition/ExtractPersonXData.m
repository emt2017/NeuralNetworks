clc
clear all
close all

for i = 1:6;

filename = sprintf('%s%d','Person',i,'.xls');

%Round 1 Data
Round1Circle = cat(2,xlsread(filename,'Circle','A4:A303'),xlsread(filename,'Circle','C4:C303'));
Round1Triangle = cat(2,xlsread(filename,'Triangle','A4:A303'),xlsread(filename,'Triangle','C4:C303'));
Round1Right = cat(2,xlsread(filename,'Right','A4:A303'),xlsread(filename,'Right','C4:C303'));
Round1Down = cat(2,xlsread(filename,'Down','A4:A303'),xlsread(filename,'Down','C4:C303'));

%Round 2 Data
Round2Circle = cat(2,xlsread(filename,'Circle','E4:E303'),xlsread(filename,'Circle','G4:G303'));
Round2Triangle = cat(2,xlsread(filename,'Triangle','E4:E303'),xlsread(filename,'Triangle','G4:G303'));
Round2Right = cat(2,xlsread(filename,'Right','E4:E303'),xlsread(filename,'Right','G4:G303'));
Round2Down = cat(2,xlsread(filename,'Down','E4:E303'),xlsread(filename,'Down','G4:G303'));

%Round 3 Data
Round3Circle = cat(2,xlsread(filename,'Circle','I4:I303'),xlsread(filename,'Circle','K4:K303'));
Round3Triangle = cat(2,xlsread(filename,'Triangle','I4:I303'),xlsread(filename,'Triangle','K4:K303'));
Round3Right = cat(2,xlsread(filename,'Right','I4:I303'),xlsread(filename,'Right','K4:K303'));
Round3Down = cat(2,xlsread(filename,'Down','I4:I303'),xlsread(filename,'Down','K4:K303'));

%if you are not me you will have to use a different save path
 save(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/PersonX/Person_' num2str(i) '.mat']);
end 