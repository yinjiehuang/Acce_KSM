function [W, g, OutputData] = Training_SOR(Training_High, Prototype_V, W_t, Scale)
% This function implements SOR KSM

[Dim, Num] = size(Training_High);  
H = round(Num / 2);

% Set the parameters
Epcilon = -4;   
s = 1.2e0;
Data = [];
Distance_Low = zeros(Num, Num);
Distance_High = zeros(Num, Num);
for i = 1:Num
    for j = (i + 1):Num
        Distance_High(i, j) = sqrt(sum((Training_High(:, j) - Training_High(:, i)).^2));
    end
end
K = zeros(H, Num);  %Changed here
for i = 1:Num
    for j = 1:H
        K(j, i) = exp( - norm(Training_High(:, i) - Prototype_V(:, j))^2 / s);
    end
end
Training_Back_Low = W_t' * K;
Z = zeros(Num, Num);
for i = 1:Num
    for j = (i + 1):Num
        Distance_Low(i, j) = sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
        if Distance_Low(i,j) ~= 0
            Z(i, j) = Distance_High(i, j) / Distance_Low(i, j);
        end
    end
end
Ini_Step = 8000;
A = K * (diag(ones(Num, Num) * ones(Num, 1)) - ones(Num, Num)) * K';
Z = Z + Z';
B = K * (diag(Z * ones(Num, 1)) - Z) * K';
t0 = clock;
Orignal_E = sum(sum(0.5 * (Distance_Low - Distance_High).^2));
for t = 1:Ini_Step
    Yita = 1;
    OrignalETemp = sum(sum(0.5 * (Distance_Low - Distance_High).^2));
    TempDistance_Low = Distance_Low;
    TempZ = Z;
    TempNewE = OrignalETemp;
    TempWt = W_t;
    flag = 1;
    while(1)
        W_tt = W_t + Yita * (pinv(A) * B * W_t - W_t);
        Training_Back_Low = W_tt' * K;
        Z = zeros(Num, Num);
        for i = 1:Num
            for j = (i + 1):Num
                Distance_Low(i, j) = sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
                if Distance_Low(i, j) ~= 0
                    Z(i, j) = Distance_High(i, j) / Distance_Low(i, j);
                end
            end
        end
        Z = Z + Z';
        New_E = sum(sum(0.5 * (Distance_Low - Distance_High).^2));
        if ((New_E <= OrignalETemp) && (New_E <= TempNewE)) || flag == 1
                Yita = Scale * Yita;
                TempZ = Z;
                TempNewE = New_E;
                TempDistance_Low = Distance_Low;
                TempWt = W_tt;
                flag = 0;
        else
            W_t = TempWt;
            break;
        end
    end
    Distance_Low = TempDistance_Low;
    New_E = TempNewE;
    Z = TempZ;
    B = K * (diag(Z * ones(Num, 1)) - Z) * K';
    g = [(A - B) * W_t(:, 1);(A - B) * W_t(:, 2)];
    NormG = log10(norm(g, inf));
    if NormG <= Epcilon || t == Ini_Step
        Data(1, 1) = t;
        Data(2, 1) = etime(clock, t0);
        Data(3, 1) = New_E;
        break;
    end
    Original_E = New_E;
    disp(['SOR_KSM: The steps are: ', int2str(t), ' .', 'The error of the Loss function: ', num2str(log10(New_E)), '  and  ', num2str(New_E), '. Infinity norm of gradient', num2str(NormG), '  .']);
end
W = W_tt;
g = (A - B) * W;
OutputData = Data;