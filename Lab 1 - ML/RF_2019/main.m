%% Clean the environment
clear all; close all; 

%% Initialisation
init; clc;

%% 1. Data loading/generation
%% 1.1. Loading and BOW
% Select dataset among {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}
[data_train, data_test, img_tr, img_te] = getData('Caltech');
k = size(data_train, 2) - 1;
%% 1.2. Plot the training set   
figure;
vis_tr = tsne(data_train(:, 1:k),'Algorithm','barneshut','NumPCAComponents',50);
vis_tr = [vis_tr data_train(:, k+1)];
plot_data(vis_tr);

%% 1.3. Plot the test set
figure;
vis_te = tsne(data_test(:, 1:k),'Algorithm','barneshut','NumPCAComponents',50);
vis_te = [vis_te data_test(:, k+1)];
scatter(vis_te(:,1),vis_te(:,2),'.b');

%% 2. Initialization
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name}; % 10 classes

%% 3. Random forest
%% 3.1. Accuracy
param.num = 10;
param.depth = 10;    % trees depth
param.splitNum = 3; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only
param.splitfunc = 'pixel';
param.T = 1 - 1/exp(1);
% Train Random Forest
trees = growTrees(data_train, param);
% Test Random Forest on training set
testTrees_script_train;

% Print accuracy
disp(accuracy_rf);
figure;
cm = confusionmat(data_train(:,end), transpose(c));
confusionchart(cm);
disp('Press any key to continue');    
pause;

% Test Random Forest on training set
testTrees_script;

% Print accuracy
disp(accuracy_rf);
figure;
cm = confusionmat(data_test(:,end), transpose(c));
confusionchart(cm);
disp('Press any key to continue');    
pause;
% Visualise
visualise(data_train,p_rf,[],0);
disp('Press any key to continue');
pause;
%% Examples
figure('Units','normalized','Position',[.05 .1 .4 .9]);
suptitle('Examples');
correct = transpose(reshape((c == transpose(data_test(:, end))), 15, length(classList)));
pred = transpose(reshape(c, 15, length(classList)));
cnt = 1;
for l = 1:length(classList)
    cor = find(correct(l, :), 1);
    inc = find(~correct(l, :), 1);
    subFolderName = fullfile(folderName,classList{l});
    imgList = dir(fullfile(subFolderName,'*.jpg'));

    I = imread(fullfile(subFolderName,imgList(img_te{l}(cor)).name));
    s1 = subaxis(5,4,cnt,'SpacingVert',0,'MR',0);
    imshow(I);
    title(string(classList{pred(l,cor)}), "Interpreter", 'none', 'FontSize', 5);
    drawnow;  
    cnt = cnt + 1;
    I = imread(fullfile(subFolderName,imgList(img_te{l}(inc)).name));    
    s2 = subaxis(5,4,cnt,'SpacingVert',0,'MR',0);
    imshow(I);
    title(string(classList{pred(l,inc)}), "Interpreter", 'none', 'FontSize', 5);
    drawnow;    
    cnt = cnt + 1;
end
figure('Units','normalized','Position',[.05 .1 .4 .9]);
suptitle('Examples');
correct = transpose(reshape((c == transpose(data_test(:, end))), 15, length(classList)));
pred = transpose(reshape(c, 15, length(classList)));
cnt = 1;
for l = length(classList)
    cor = find(correct(l, :), 1);
    inc = find(~correct(l, :), 1);
    subFolderName = fullfile(folderName,classList{l});
    imgList = dir(fullfile(subFolderName,'*.jpg'));

    I = imread(fullfile(subFolderName,imgList(img_te{l}(cor)).name));
    s1 = subaxis(1,2,cnt,'SpacingVert',0,'MR',0);
    imshow(I);
    title(string(classList{pred(l,cor)}), "Interpreter", 'none', 'FontSize', 5);
    drawnow;  
    cnt = cnt + 1;
    I = imread(fullfile(subFolderName,imgList(img_te{l}(inc)).name));    
    s2 = subaxis(1,2,cnt,'SpacingVert',0,'MR',0);
    imshow(I);
    title(string(classList{pred(l,inc)}), "Interpreter", 'none', 'FontSize', 5);
    drawnow;    
    cnt = cnt + 1;
end
%% 4. Random forest parameters
%% 4.1 Number of trees
for T = [0.9] % Number of trees, try {1,3,5,10, or 20}
    param.num = 30;
    param.depth = 7;    % trees depth
    param.splitNum = 3; % Number of trials in split function
    param.split = 'IG'; % Currently support 'information gain' only
    param.splitfunc = 'axis';
    param.T = T;
    % Train Random Forest
    tic
    trees = growTrees(data_train, param);
    toc
    % Test on train
    testTrees_script_train;
    % Print accuracy
    figure;
    disp(accuracy_rf);
    cm = confusionmat(data_test(:,end), c);
    confusionchart(cm);
%     disp('Press any key to continue');    
%     pause;    
    % Test Random Forest
    tic
    testTrees_script;
    toc
    % Print accuracy
    figure;
    disp(accuracy_rf);
    cm = confusionmat(data_test(:,end), c);
    confusionchart(cm);
%     disp('Press any key to continue');    
%     pause;
end
%% 4.2 Depth of trees
for N = [2, 5, 7, 11] % Tree depth, try {2,5,7,11}
    param.num = 10;
    param.depth = N;    % trees depth
    param.splitNum = 10; % Number of trials in split function
    param.split = 'IG'; % Currently support 'information gain' only
    param.splitfunc = 'axis';
    param.T = 1 - 1/exp(1);    
    % Train Random Forest
    tic
    trees = growTrees(data_train, param);
    toc
    % Test Random Forest
    tic
    testTrees_script;
    toc
    % Print accuracy
    disp(accuracy_rf);
    cm = confusionmat(data_test(:,end), c);
    figure
    confusionchart(cm);
    disp('Press any key to continue');    
    pause;
    % Visualise
    visualise(data_train,p_rf,[],0);
    disp('Press any key to continue');
    pause;
    
    
end