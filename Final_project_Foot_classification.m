%% Final project Foot classification *script*
clc
close all
clear 

%% Load data 
%% path selection and searching for files with VP
seleceted_path = uigetdir;
cd(seleceted_path);
folder_list_left = dir('*-L.jpg');  % left foot 
for i = 1:length(folder_list_left)
folder_list_left(i) = folder_list_left(i).name  ;
end 
folder_list_right = dir('*-R.jpg'); % right foot 

imread(folder_list_left.name(1))