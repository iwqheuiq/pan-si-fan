function v_poincare = exp_map(v, K)
    if K >= 0
        error('曲率 k 必须是负值。');
    end
    norm_v = norm(v);
    if norm_v == 0
        v_poincare = zeros(size(v));  % 如果 v 是零向量，则映射结果也是零向量
    else
        scaleFactor = tanh(sqrt(-K) * norm_v) / (sqrt(-K) * norm_v);
        v_poincare = scaleFactor * v;
    end
end
