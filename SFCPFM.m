clc;
tic
close all
I0 = imread("E:\图\第二篇\11\94.jpg"); % 替换为你的图片路径
 img= rgb2gray(I0); % 转换为灰度图像，如果已经是灰度，则忽略这步
img = im2double(img); % 转换为双精度格式
G=img;
% 添加高斯噪声，均值为0，方差为0.1
% 读取图像
 % G = imnoise(img, 'gaussian', 0, 0.15); % 使用 imnoise 添加高斯噪声
 % G=imnoise(img,'salt & pepper',0.1);
 % G = imnoise(img, 'speckle', 0.5);
figure, imshow(G);
imwrite(G, "E:\图\noisy_image.png"); % 保存带噪声的图像
%%
% img=G;
C =3; % 聚类数目
iterations = 20; % 迭代次数
K = -1; % 庞加莱曲率
m =2; % 模糊指数
lambda=0.3;
sigma=2;
r =1;
% 图像填充和数据转换
padded_img = padarray(G, [1 1], 0, 'both'); 
[rows, cols] = size(padded_img);
data = zeros((rows-2)*(cols-2), 9);
index = 1;

for j = 2:cols-1
    for i = 2:rows-1
        neighborhood = padded_img(i-1:i+1, j-1:j+1);
        data(index, :) = neighborhood(:)';
        index = index + 1;
    end
end

% 映射到庞加莱球上并进行聚类
data = (data - min(data)) ./ (max(data) - min(data));
mapped_data = exp_map(data, K);
iniCentroids = mapped_data(randperm(size(mapped_data, 1), C), :);

% 执行聚类
% [Idx, U, V] = Spoincare_FCM(mapped_data, iniCentroids, C, iterations,m,K,lambda);

%%
alpha = 0.1;        % 曲率参数
k = 2;            % Filtration保留数量
T = 100;          % 最大迭代次数
epsilon = 1e-5;   % 收敛阈值
K=-1;

 [centers, U,t] = HypeFCM(mapped_data, C, m, alpha, k, T, epsilon);
disp(t)
 [~, Idx] = max(U, [], 2); 
 %%
% [Idx, U, V] = poincare_KMeans(mapped_data, C, iniCentroids, iterations,K);
 % [Idx, U, V] = Kpoincare_FCM(mapped_data,C, iniCentroids, iterations,m,K,sigma);
% [Idx, U, V] = KSpoincare_FCM(mapped_data,C, iniCentroids, iterations,m,K,sigma,lambda);
% 将聚类结果转换为图像
clusteredImage = reshape(Idx, [rows-2, cols-2]);
fs = Label_image(img, clusteredImage);
figure;
imshow(fs);
imwrite(uint8(fs), "E:\图\SFCPFM.png");
toc;