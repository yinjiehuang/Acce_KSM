function IniWeight(H)
% This function is used to initialize the weight matrix
% Author: Yinjie Huang

Weight = zeros(H,200);

for i = 1:100
    Weight(:, ((i - 1) * 2 + 1):((i - 1) * 2 + 2)) = rand(H, 2);
end

save Weight;