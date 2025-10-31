function [Idx,U, centroids]= poincare_FCM(mapped_data, iniCentroids,C,iterations,m, K)
    [numOfData, ~] = size(mapped_data);
    centroids = iniCentroids; % 使用初始质心
    U = rand(numOfData, C); % 初始化模糊矩阵
    U = U ./ sum(U, 2); % 按行归一化

    % 设定隶属度和聚类中心的变化阈值
    threshold_centroids = 1e-5;
  
    for iter = 1:iterations
         centroids_prev = centroids;
         U = update_membership(U, mapped_data, centroids_prev, m, K); 
        % 更新聚类中心
         centroids = update_centroids(U, mapped_data,centroids_prev,m, K);
        % 检查聚类中心的变化
        delta_centroids = max(max(abs(centroids - centroids_prev)));
        % 如果隶属度和聚类中心变化都小于阈值，则停止迭代
        if  delta_centroids < threshold_centroids
            break;
        end
    end

    % 计算最终的隶属度
[~, Idx] = max(U, [], 2);  % 转换U成为Idx
end


function U = update_membership(U,mapped_data, centroids, m, K)
    epsilon = 1e-10;  % 可根据具体数值范围调整
    [numOfData, C] = size(U);
    for i = 1:numOfData
        for j = 1:C
            dist_to_j = (1/(poincare_distance(mapped_data(i, :), centroids(j, :), K)+epsilon))^(2/(m-1));
            dist_sum = 0 ;
            for k = 1:C
                dist_sum =dist_sum+(1/(poincare_distance(mapped_data(i, :), centroids(k, :), K)+epsilon))^(2/(m-1));
                % 防止分母为0           
            end
            U(i, j) = dist_to_j / dist_sum;
        end
    end

end

function centroids = update_centroids(U, mapped_data,~, m, K)
    [~, C] = size(U);
    dim = size(mapped_data, 2);
    centroids = zeros(C, dim);
  
    % 对每个质心进行更新
    for j = 1:C
        u_m = U(:, j) .^ m; % 隶属度的 m 次幂
        centroids(j, :) = frechet(mapped_data,u_m , K, 100, 1e-6, 1e-6);
    end

end


% frechet均值
function mu = frechet(X, w, K, max_iter, rtol, atol)
    mu = X(1,:);
    
    for iter = 1:max_iter
        muPrev = mu;
        mu_ss = sum(mu.^2, 2);
        x_ss = sum(X.^2, 2);
       
        xmu_ss = sum((X - mu).^2, 2);
        
        www=(-K * xmu_ss ./ ((1 + K * x_ss) .* (1 + K * mu_ss))).*(1./(1 + K * x_ss)+0.00001);
        
        alphas = l_prime(www);
        
       
        alphas = alphas .* w;
       
        c = sum(alphas .* x_ss);
        
        b = sum(bsxfun(@times, alphas, X), 1); % 权重乘以数据点
         a = sum(alphas);
        
        b_ss = sum(b.^2);
        d = ((a - K * c).^2 + 4 * K * b_ss);
        if any(d < 0)
            d(d < 0) = 0;
        end
        eta = (a - K * c - sqrt(d)) ./ (2 * (-K) * b_ss);
        
        mu = eta .* b; % 一维数组乘以一维数组
        dist = sqrt(sum((mu - muPrev).^2));
        prev_dist = sqrt(sum(muPrev.^2));
        if all(dist < atol) || all(dist ./ prev_dist < rtol)
            break;
        end
    end
end

function ret = l_prime(y)
    cond = y < 1e-12;
    val = 4 * ones(size(y));
    y(y < 0) = 0; % 确保acosh的参数大于等于1
    ret = 2 * acosh(1 + 2 * y) ./ sqrt(y.^2 + y);
    ret(cond) = val(cond);
end


function d = poincare_distance(x, y, K)
    epsilon = 1e-10;
    norm_x = sqrt(sum(x.^2, 2));
    norm_y = sqrt(sum(y.^2, 2));
    numerator = sum((x - y).^2, 2);
    denominator = max((1 + K * (norm_x.^2)) .* (1 + K * (norm_y.^2)), epsilon);
    lambda = 1 - 2 * K * numerator ./ denominator;
    % 保证acosh的参数不小于1
    lambda = max(lambda, 1 + epsilon);
    d =  1./sqrt(-K)*acosh(lambda);
end