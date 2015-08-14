function [W, g, OutputData] = Training_KSM(Training_High, Prototype_V, W_t)
%This function use the IM Iteration

[Dim,Num] = size(Training_High);  %Dim is the dimension of the high dataset
H = round(Num / 2);

% Set some parameters
Epcilon = -4;  
s = 1.2e0;
Data = [];
%compute the distance of lower and higher dimension
Distance_Low = zeros(Num, Num);
Distance_High = zeros(Num, Num);
% This is the matrix of the High dimension Eucildean Distance
for i = 1:Num
    for j = (i + 1):Num
        Distance_High(i, j) = sqrt(sum((Training_High(:, j) - Training_High(:, i)).^2));
    end
end

%Here is the distance matrix of Low dimensional space for the starting step
K = zeros(H, Num);  %Changed here
for i = 1:Num
    for j = 1:H
        K(j,i) = exp(- norm(Training_High(:, i) - Prototype_V(:, j))^2/s);
    end
end
Training_Back_Low = W_t' * K;
Z = zeros(Num, Num);
for i = 1:Num
    for j = (i + 1):Num
        Distance_Low(i, j) = sqrt(sum((Training_Back_Low(:, j) - Training_Back_Low(:, i)).^2));
        if Distance_Low(i, j) ~= 0
            Z(i, j) = Distance_High(i, j) / Distance_Low(i, j);
        end
    end
end
Ini_Step = 80000;
% The definition of matrix A and B, please refer to the paper
A = K * (diag(ones(Num, Num) * ones(Num, 1)) - ones(Num, Num)) * K';
Z = Z + Z';
B = K * (diag(Z * ones(Num, 1)) - Z) * K';
ErrorLine = [];
t0 = clock;   
Orignal_E = sum(sum(0.5 * (Distance_Low - Distance_High).^2));
for t = 1:Ini_Step
    W_tt = pinv(A) * B * W_t;
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
    B = K * (diag(Z * ones(Num, 1)) - Z) * K';
    New_E = sum(sum(0.5 * (Distance_Low - Distance_High).^2));
    g = [(A - B) * W_tt(:, 1);(A - B) * W_tt(:, 2)];
    NormG = log10(norm(g, inf));
    if  NormG <= Epcilon || t==Ini_Step
        % Store the result
        Data(1, 1) = t;
        Data(2, 1) = etime(clock, t0);
        Data(3, 1) = New_E;
        break;
    end
    W_t = W_tt;
    Original_E = New_E;
    disp(['IM_KSM: The steps are: ', int2str(t), ' .', 'The error of the Loss function: ', num2str(log10(New_E)), '  and  ', num2str(New_E), '. Infinity norm of gradient', num2str(NormG),'  .']);
end

W = W_tt;
g = (A - B) * W;
OutputData = Data;