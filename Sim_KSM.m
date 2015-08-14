% This script runs KSM and its two accelerated algorithms
% Author: Yinjie Huang
% The work has been published in IJCNN 2011
% Reference:
% Yinjie Huang, Michael Georgiopoulos, Georgios C. Anagnostopoulos,
% Accelerated Learning of Generalized Sammon Mappings in IJCNN 2011

% First of all let's define the data structure
% All the results are stored in this structure
clear;
clc;

Result.Datatype = [];
Result.NumTraining = [];
Result.IniWeight = [];
Result.IM.time = [];
Result.IM.iteration = [];
Result.IM.error = [];
Result.IM.g = [];
Result.IM.FinWeight = [];
Result.SOR.time = [];
Result.SOR.iteration = [];
Result.SOR.error = [];
Result.SOR.g = [];
Result.SOR.FinWeight = [];
Result.PARTAN.time = [];
Result.PARTAN.iteration = [];
Result.PARTAN.error = [];
Result.PARTAN.g = [];
Result.PARTAN.FinWeight = [];


% Load the data
load SwissRData75
Training_High = SwissRData;

% If necessary, you can normalize the data
% Training_High=Training_High/max(max(Training_High));%Normalize the Dataset
    
[Dim, Num] = size(Training_High);
filename = 'SwissRoll';

Result.Datatype = 'SwissRoll';
Result.NumTraining = Num;
H = round(Num / 2);

% Initilize the weight
IniWeight(H);
load Weight;
Result.IniWeight = Weight;

save(filename, 'Result');
RAND = randperm(Num);
% Define the prototypes
Prototype_V = Training_High(:,RAND(1:H));

for trail = 1:100
    fprintf('%s%d%s\n', '*************** Trail ', 100, ' ***************');
    W_t = Weight(:, ((trail - 1) * 2 + 1):((trail - 1) * 2 + 2)) * 50;
    
    disp(['******************************* Training original KSM *******************************']);
    [W, g, IMData] = Training_KSM(Training_High, Prototype_V, W_t);
    load(filename);
    %Update data
    Result.IM.time = [Result.IM.time, IMData(1, :)];
    Result.IM.iteration = [Result.IM.iteration, IMData(2, :)];
    Result.IM.error = [Result.IM.error, IMData(3, :)];
    Result.IM.g = [Result.IM.g, g];
    Result.IM.FinWeight = [Result.IM.FinWeight, W];
    save(filename, 'Result');
    
    disp(['******************************* Training SOR KSM *******************************']);
    [W, g, SORData] = Training_SOR(Training_High, Prototype_V, W_t, 2.0);
    %Update data
    Result.SOR.time = [Result.SOR.time, SORData(1, :)];
    Result.SOR.iteration = [Result.SOR.iteration, SORData(2, :)];
    Result.SOR.error = [Result.SOR.error, SORData(3, :)];
    Result.SOR.g = [Result.SOR.g, g];
    Result.SOR.FinWeight = [Result.SOR.FinWeight, W];
    save(filename, 'Result');
    
    disp(['******************************* Training Partan KSM *******************************']);
    [W, g, PARTANData] = Training_Partan(Training_High, Prototype_V, W_t, 2.0);
    %Update data
    Result.PARTAN.time = [Result.PARTAN.time, PARTANData(1, :)];
    Result.PARTAN.iteration = [Result.PARTAN.iteration, PARTANData(2, :)];
    Result.PARTAN.error = [Result.PARTAN.error, PARTANData(3, :)];
    Result.PARTAN.g = [Result.PARTAN.g, g];
    Result.PARTAN.FinWeight = [Result.PARTAN.FinWeight, W];
    save(filename, 'Result');
end