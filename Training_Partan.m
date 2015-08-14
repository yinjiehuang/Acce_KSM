function [W, g, OutputData] = Training_Partan(Training_High, Prototype_V, W_t, Scale)
% This function implements Partan KSM

[Dim, Num] = size(Training_High);  
H = round(Num / 2);

% Set some parameters
Epcilon = -4;   
s = 1.2e0;
Data = [];
W_t_1 = W_t;
Distance_Lowk_1 = zeros(Num, Num);
Distance_Lowk = zeros(Num, Num);
Distance_Lowk_2 = zeros(Num, Num);
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

A = K * (diag(ones(Num, Num) * ones(Num, 1)) - ones(Num, Num)) * K';
PivA = pinv(A);
Training_Back_Low = W_t_1' * K;
Zk_1 = zeros(Num, Num);
for i = 1:Num
    for j = (i + 1):Num
        Distance_Lowk_1(i, j) = sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
        if Distance_Lowk_1(i, j) ~= 0
            Zk_1(i, j) = Distance_High(i, j) / Distance_Lowk_1(i, j);
        end
    end
end
Zk_1 = Zk_1 + Zk_1';
Bk_1 = K * (diag(Zk_1 * ones(Num, 1)) - Zk_1) * K';
gk_1 = [(A -Bk_1) * W_t_1(:, 1);(A - Bk_1) * W_t_1(:, 2)];

W_t = PivA * Bk_1 * W_t_1;
Training_Back_Low = W_t' * K;
Zk = zeros(Num, Num);
for i = 1 : Num
    for j = (i + 1) : Num
        Distance_Lowk(i, j) = sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
        if Distance_Lowk(i, j) ~= 0
            Zk(i, j) = Distance_High(i, j) / Distance_Lowk(i, j);
        end
    end
end
Zk = Zk + Zk';
Bk = K * (diag(Zk * ones(Num, 1)) - Zk) * K';
Ini_Step = 3000;   
t0 = clock;
for t = 1:Ini_Step
    g = [(A - Bk) * W_t(:, 1);(A - Bk) * W_t(:, 2)];
    NormG = log10(norm(g, inf));
    if NormG <= Epcilon || t == Ini_Step
        Data(1, 1) = t;
        Data(2, 1) = etime(clock, t0);
        Data(3, 1) = New_E;
        break;
    end
    JDistance_Low = Distance_Lowk;
    JZ = Zk;
    JB = Bk;
    Alpha = 1;
    OrignalETemp = sum(sum(0.5 * (JDistance_Low - Distance_High).^2));
    TempDistance_Low = Distance_Lowk;
    TempWt = W_t;
    TempZ = Zk;
    TempNewE = OrignalETemp;
    flag1 = 1;
    while(1)
        J = W_t + Alpha * (PivA * Bk * W_t - W_t);
        Training_Back_Low = J' * K;
        JZ = zeros(Num, Num);
        for i = 1:Num
            for j = (i + 1):Num
                JDistance_Low(i, j)=sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
                if JDistance_Low(i, j) ~= 0
                    JZ(i, j) = Distance_High(i, j) / JDistance_Low(i, j);
                end
            end
        end
        New_E = sum(sum(0.5 * (JDistance_Low - Distance_High).^2));
        if (New_E <= OrignalETemp && New_E <= TempNewE) || flag1 == 1
            Alpha = Scale * Alpha;
            TempZ = JZ + JZ';
            TempNewE = New_E;
            TempDistance_Low = JDistance_Low;
            TempWt = J;
            flag1 = 0;
        else
            J = TempWt;
            break;
        end
    end
    JDistance_Low = TempDistance_Low;
    New_E = TempNewE;
    JZ = TempZ;
    JB = K * (diag(JZ * ones(Num, 1)) - JZ) * K';
    d = J - W_t_1;
    temptemp = gk_1' * [d(:, 1);d(:, 2)];
    if temptemp >= 0
        OrignalETemp = New_E;
        TempZ = Zk_2;
        TempNewE = OrignalETemp;
        TempDistance_Low = Distance_Lowk_2;
        TempWt = J;
        Beta = 1;
        flag2 = 1;
        while(1)
            W_tt = J + Beta * (PivA * JB * J - J);
            Training_Back_Low = W_tt' * K;
            Zk_2 = zeros(Num, Num);
            for i = 1:Num
                for j = (i + 1):Num
                    Distance_Lowk_2(i, j) = sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
                    if Distance_Lowk_2(i, j) ~= 0
                        Zk_2(i, j) = Distance_High(i, j) / Distance_Lowk_2(i, j);
                    end
                end
            end
            New_E = sum(sum(0.5 * (Distance_Lowk_2 - Distance_High).^2));
            if (New_E <= OrignalETemp && New_E <= TempNewE) || flag2 == 1
                Beta = Scale * Beta;
                TempZ = Zk_2 + Zk_2';
                TempNewE = New_E;
                TempDistance_Low = Distance_Lowk_2;
                TempWt = W_tt;
                flag2 = 0;
            else
                W_tt = TempWt;
                break;
            end
        end
        Distance_Lowk_2 = TempDistance_Low;
        New_E = TempNewE;
        Zk_2 = TempZ;
        Bk_2 = K * (diag(Zk_2 * ones(Num, 1)) - Zk_2) * K';
    else
        %We got to use the PARTAN-IM
        Distance_Lowk_2 = Distance_Lowk_1;
        Zk_2 = Zk_1;
        Bk_2 = Bk_1;
        OrignalETemp = sum(sum(0.5 * (Distance_Lowk_2 - Distance_High).^2));
        TempZ = Zk_1;
        TempNewE = OrignalETemp;
        TempDistance_Low = Distance_Lowk_1;
        TempWt = W_t_1;
        Beta = 1;
        flag3 = 1;
        while(1)
            W_tt = W_t_1 + Beta * d;
            Training_Back_Low = W_tt' * K;
            Zk_2 = zeros(Num, Num);
            for i = 1:Num
                for j = (i + 1):Num
                    Distance_Lowk_2(i, j) = sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
                    if Distance_Lowk_2(i, j) ~= 0
                        Zk_2(i, j) = Distance_High(i, j) / Distance_Lowk_2(i, j);
                    end
                end
            end
            New_E = sum(sum(0.5 * (Distance_Lowk_2 - Distance_High).^2));
            if (New_E <= OrignalETemp && New_E <= TempNewE) || flag3 == 1
                Beta = Scale * Beta;
                TempZ = Zk_2 + Zk_2';
                TempNewE = New_E;
                TempDistance_Low = Distance_Lowk_2;
                TempWt = W_tt;
                flag3 = 0;
            else
                W_tt = TempWt;
                break;
            end
        end
        Distance_Lowk_2 = TempDistance_Low;
        New_E = TempNewE;
        Zk_2 = TempZ;
        Bk_2 = K * (diag(Zk_2 * ones(Num, 1)) - Zk_2) * K';  
    end
    gk_1 = g;
    W_t_1 = W_t;
    Distance_Lowk_1 = Distance_Lowk;
    Zk_1 = Zk;
    Bk_1 = Bk;
    W_t = W_tt;
    Distance_Lowk = Distance_Lowk_2;
    Zk = Zk_2;
    Bk = Bk_2;
    Original_E = New_E;
    disp(['PARTAN_KSM: The steps are: ', int2str(t), ' .', 'The error of the Loss function: ', num2str(log10(New_E)), '  and  ', num2str(New_E), '. Infinity norm of gradient', num2str(NormG), '  .']);
end
W = W_tt;
g = (A - Bk) * W;
OutputData = Data;