%%
% 定义文件位置
digitDatasetPath = uigetdir(path);
%文件作为图片，并保留子文件夹
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%% 
%训练于验证量随机分配
numberoftrains = 25;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numberoftrains,'randomize');
%%
%图片为500*500的rgb图片则为三维矩阵，并仅作两组分类
inputsize = [500 500 3];
numclass = 2;
%%
% 开始创建CNN
layers = [
    imageInputLayer(inputsize)%图像输入层
    convolution2dLayer(10,10)%10个卷积核，大小为10*10
    batchNormalizationLayer%归一化层，加快收敛
    reluLayer%修正参数
    fullyConnectedLayer(numclass)%全连接层，使其受到权重影响
    softmaxLayer%归一化指数
    classificationLayer];%分类输出
%%
% 设置训练采用动量随机梯度下降
% 最大训练轮次为20次
% 学习率为0.0001
%验证源为验证图片
%验证频率为40
% 不显示进度信息
% 描绘图像
options = trainingOptions("sgdm","MaxEpochs",20, ...
    "InitialLearnRate",1e-4, ...
    'ValidationData',imdsValidation,...
    'VerboseFrequency',40, ...
    "Verbose",false,...
    "Plots",'training-progress');
%%
%整理整个神经网络
net = trainNetwork(imdsTrain,layers,options);
%%
% 开始进行分类新图片
questionimagePath = uigetdir(path);%待分类图片位置
cityimagePath = uigetdir(path);%分类后的城市图片的位置
treeimagePath = uigetdir(path);%分类后的树的图片的位置
allquestionimages = dir([questionimagePath,'\*.jpg']);%获取待分类图片
cd exampledata\question\
%%
for i = 1:length(allquestionimages)%一个一个按net分类
    questionimage = allquestionimages(i).name;
    Pimage = imread(fullfile(questionimagePath,questionimage));
    Aimage = imresize(Pimage, [500 500]);
    [Class,rate] = classify(net,Aimage);%得到类别与相似度
    if Class == 'City'
        copyfile(questionimage,cityimagePath); 
    else
        copyfile(questionimage,treeimagePath);
    end
end