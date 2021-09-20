%ASC Student Simulation Challenge 2021
%Rowan University Submission

%Prep workspace
clear all;
clc;

%%DNN-Regression Model
%Import and partition data
dataset = xlsread('ASC Student Challenge 2021_Data Set',...
    'ASC 2021  Challenge Data Set');

%Read and Label Input Columns
Cycle_Number = dataset(:,1);
Heat_Rate_1 = dataset(:,2);
Ramp_1_Duration = dataset(:,3);
Temperature_Dwell_1 = dataset(:,4);
Heat_Rate_2 = dataset(:,5);
Ramp_2_Duration = dataset(:,6);
Temperature_Dwell_2 = dataset(:,7);
Vacuum_Pressure = dataset(:,8);
Vacuum_Start_Time = dataset(:,9);
Vacuum_Duration = dataset(:,10);
Autoclave_Pressure = dataset(:,11);
Autoclave_Start_Time = dataset(:,12);
Autoclave_Duration = dataset(:,13);

%Read and Label Output Columns
AD_Porosity = dataset(:,16);
PR_Porosity = dataset(:,17);
Eff_Porosity = dataset(:,18);
Max_Fiber_vol_fraction = dataset(:,19);
Cure_Cycle_Total_Time = dataset(:,20);
AD_Volume = dataset(:,21);
PR_Volume = dataset(:,22);

%Set input matrix
X_i = [Cycle_Number  Heat_Rate_1 Ramp_1_Duration Temperature_Dwell_1... 
     Heat_Rate_2 Ramp_2_Duration Temperature_Dwell_2 Vacuum_Pressure... 
     Vacuum_Start_Time Vacuum_Duration Autoclave_Pressure Autoclave_Start_Time... 
     Autoclave_Duration];
 
Y_i = [AD_Porosity PR_Porosity Eff_Porosity Max_Fiber_vol_fraction...
     Cure_Cycle_Total_Time AD_Volume PR_Volume];

%Shuffle Data to improve generalizablity
XY = [X_i Y_i];   %Combine X & Y to reshuffle
Shuffle_row_XY = XY(randperm(size(XY,1)),:); %Reshuffle Rows
Shuffle_row_XY(isnan(Shuffle_row_XY))=0; %If Nan exists set to zero
X = Shuffle_row_XY(:,1:13);
Y = Shuffle_row_XY(:,14:20);

%Partition into Training and Testing data
XTrain = X(3:1262 ,2:13);  %Train with 70% of the samples
YTrain = Y(3:1262, 1:7);   %For Clarificiation (Row start,Row end);(Col start, Col end)

XValidation = X(1263:1802, 2:13);
YValidation = Y(1263:1802, 1:7);

layers = [ ...
    featureInputLayer(12, 'Name', 'Input')    %Corresponds to the number of inputs from xlsx
    fullyConnectedLayer(10, 'Name', 'FC_1')    
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','ReLu_1')
    fullyConnectedLayer(9,'Name', 'FC_2')    
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','ReLu_2')
    fullyConnectedLayer(7,'Name', 'FC_3')
    regressionLayer('Name','Output')  
    ];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',5, ...
    'GradientDecayFactor',0.89, ...
    'L2Regularization', 0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'MaxEpochs',10, ... 
    'MiniBatchSize',16, ...
    'Plots','training-progress'...
    );


net = trainNetwork(XTrain,YTrain,layers,options);

YPred = predict(net,XValidation);

rmse = sqrt(mean((YValidation - YPred).^2));

%Plot DNN line chart
lgraph = layerGraph(layers);
figure(1)
plot(lgraph)

%%Identify Process parameters with DNN model%%
%Extract weights and biases for optimal paths and cost function 

Weight_hidden1 = net.Layers(2).Weights;
Bias_hidden1 = net.Layers(2).Bias;

Weight_hidden2 = net.Layers(5).Weights;
Bias_hidden2 = net.Layers(5).Bias;

Weight_toOut = net.Layers(8).Weights;
Bias_toOut = net.Layers(8).Bias;

%Extract sizes for computations
[i,j] = size(Weight_hidden1); %[Row,Col]
[k,l] = size(Weight_hidden2); %[Row,Col]
[m,n] = size(Weight_toOut); %[Row,Col]

%Extract values for paths
[max_H1, row_locmax_H1] = max(Weight_hidden1);
[min_H1, row_locmin_H1] = min(Weight_hidden1);
[max_H2, row_locmax_H2] = max(Weight_hidden2);
[min_H2, row_locmin_H2] = min(Weight_hidden2);
[max_Hout, row_locmax_Hout] = max(Weight_toOut);
[min_Hout, row_locmin_Hout] = min(Weight_toOut);

%Note most of the data analysis can be simplified with loops, in some
%cases, indexing or matrix multiplication complicated this, which is why it
%is in the present form.

%%%%Feature Path (1)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 1 (Max)
Input1_pathVal_max_Layer1 = max_H1(1)+ max_H2(row_locmax_H1(1))+ max_Hout(row_locmax_H2(row_locmax_H1(1)));
NeuronsMat_Feature_1_Max = [1              row_locmax_H1(1)     row_locmax_H2(row_locmax_H1(1));
                       row_locmax_H1(1)    row_locmax_H2(row_locmax_H1(1))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(1)))
];
%Path for Feature 1 (Min)
Input1_pathVal_min_Layer1 = min_H1(1)+ min_H2(row_locmin_H1(1))+ min_Hout(row_locmin_H2(row_locmin_H1(1)));
NeuronsMat_Feature_1_min = [1              row_locmin_H1(1)     row_locmin_H2(row_locmin_H1(1));
                       row_locmin_H1(1)    row_locmin_H2(row_locmin_H1(1))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(1)))
];

%%%%Feature Path (2)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 2 (Max)
Input2_pathVal_max_Layer1 = max_H1(2)+ max_H2(row_locmax_H1(2))+ max_Hout(row_locmax_H2(row_locmax_H1(2)));
NeuronsMat_Feature_2_Max = [2              row_locmax_H1(2)     row_locmax_H2(row_locmax_H1(2));
                       row_locmax_H1(2)    row_locmax_H2(row_locmax_H1(2))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(2)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 2 (Min)
Input2_pathVal_min_Layer1 = min_H1(2)+ min_H2(row_locmin_H1(2))+ min_Hout(row_locmin_H2(row_locmin_H1(2)));
NeuronsMat_Feature_2_min = [2              row_locmin_H1(2)     row_locmin_H2(row_locmin_H1(2));
                       row_locmin_H1(2)    row_locmin_H2(row_locmin_H1(2))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(2)))
];

%%%%Feature Path (3)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 3 (Max)
Input3_pathVal_max_Layer1 = max_H1(3)+ max_H2(row_locmax_H1(3))+ max_Hout(row_locmax_H2(row_locmax_H1(3)));
NeuronsMat_Feature_3_Max = [3              row_locmax_H1(3)     row_locmax_H2(row_locmax_H1(3));
                       row_locmax_H1(3)    row_locmax_H2(row_locmax_H1(3))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(3)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 3 (Min)
Input3_pathVal_min_Layer1 = min_H1(3)+ min_H2(row_locmin_H1(3))+ min_Hout(row_locmin_H2(row_locmin_H1(3)));
NeuronsMat_Feature_3_min = [3              row_locmin_H1(3)     row_locmin_H2(row_locmin_H1(3));
                       row_locmin_H1(3)    row_locmin_H2(row_locmin_H1(3))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(3)))
];

%%%%Feature Path (4)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 4 (Max)
Input4_pathVal_max_Layer1 = max_H1(4)+ max_H2(row_locmax_H1(4))+ max_Hout(row_locmax_H2(row_locmax_H1(4)));
NeuronsMat_Feature_4_Max = [4              row_locmax_H1(4)     row_locmax_H2(row_locmax_H1(4));
                       row_locmax_H1(4)    row_locmax_H2(row_locmax_H1(4))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(4)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 4 (Min)
Input4_pathVal_min_Layer1 = min_H1(4)+ min_H2(row_locmin_H1(4))+ min_Hout(row_locmin_H2(row_locmin_H1(4)));
NeuronsMat_Feature_4_min = [4              row_locmin_H1(4)     row_locmin_H2(row_locmin_H1(4));
                       row_locmin_H1(4)    row_locmin_H2(row_locmin_H1(4))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(4)))
];

%%%%Feature Path (5)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 5 (Max)
Input5_pathVal_max_Layer1 = max_H1(5)+ max_H2(row_locmax_H1(5))+ max_Hout(row_locmax_H2(row_locmax_H1(5)));
NeuronsMat_Feature_5_Max = [5              row_locmax_H1(5)     row_locmax_H2(row_locmax_H1(5));
                       row_locmax_H1(5)    row_locmax_H2(row_locmax_H1(5))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(5)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 5 (Min)
Input5_pathVal_min_Layer1 = min_H1(5)+ min_H2(row_locmin_H1(5))+ min_Hout(row_locmin_H2(row_locmin_H1(5)));
NeuronsMat_Feature_5_min = [5              row_locmin_H1(5)     row_locmin_H2(row_locmin_H1(5));
                       row_locmin_H1(5)    row_locmin_H2(row_locmin_H1(5))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(5)))
];

%%%%Feature Path (6)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 6 (Max)
Input6_pathVal_max_Layer1 = max_H1(6)+ max_H2(row_locmax_H1(6))+ max_Hout(row_locmax_H2(row_locmax_H1(6)));
NeuronsMat_Feature_6_Max = [6              row_locmax_H1(6)     row_locmax_H2(row_locmax_H1(6));
                       row_locmax_H1(6)    row_locmax_H2(row_locmax_H1(6))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(6)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 6 (Min)
Input6_pathVal_min_Layer1 = min_H1(6)+ min_H2(row_locmin_H1(6))+ min_Hout(row_locmin_H2(row_locmin_H1(6)));
NeuronsMat_Feature_6_min = [6              row_locmin_H1(6)     row_locmin_H2(row_locmin_H1(6));
                       row_locmin_H1(6)    row_locmin_H2(row_locmin_H1(6))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(6)))
];

%%%%Feature Path (7)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 7 (Max)
Input7_pathVal_max_Layer1 = max_H1(7)+ max_H2(row_locmax_H1(7))+ max_Hout(row_locmax_H2(row_locmax_H1(7)));
NeuronsMat_Feature_7_Max = [7              row_locmax_H1(7)     row_locmax_H2(row_locmax_H1(7));
                       row_locmax_H1(7)    row_locmax_H2(row_locmax_H1(7))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(7)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 7 (Min)
Input7_pathVal_min_Layer1 = min_H1(7)+ min_H2(row_locmin_H1(7))+ min_Hout(row_locmin_H2(row_locmin_H1(7)));
NeuronsMat_Feature_7_min = [7              row_locmin_H1(7)     row_locmin_H2(row_locmin_H1(7));
                       row_locmin_H1(7)    row_locmin_H2(row_locmin_H1(7))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(7)))
];

%%%%Feature Path (8)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 8 (Max)
Input8_pathVal_max_Layer1 = max_H1(8)+ max_H2(row_locmax_H1(8))+ max_Hout(row_locmax_H2(row_locmax_H1(8)));
NeuronsMat_Feature_8_Max = [8              row_locmax_H1(8)     row_locmax_H2(row_locmax_H1(8));
                       row_locmax_H1(8)    row_locmax_H2(row_locmax_H1(8))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(8)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 8 (Min)
Input8_pathVal_min_Layer1 = min_H1(8)+ min_H2(row_locmin_H1(8))+ min_Hout(row_locmin_H2(row_locmin_H1(8)));
NeuronsMat_Feature_8_min = [8              row_locmin_H1(8)     row_locmin_H2(row_locmin_H1(8));
                       row_locmin_H1(8)    row_locmin_H2(row_locmin_H1(8))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(8)))
];

%%%%Feature Path (9)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 9 (Max)
Input9_pathVal_max_Layer1 = max_H1(9)+ max_H2(row_locmax_H1(9))+ max_Hout(row_locmax_H2(row_locmax_H1(9)));
NeuronsMat_Feature_9_Max = [9              row_locmax_H1(9)     row_locmax_H2(row_locmax_H1(9));
                       row_locmax_H1(9)    row_locmax_H2(row_locmax_H1(9))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(9)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 9 (Min)
Input9_pathVal_min_Layer1 = min_H1(9)+ min_H2(row_locmin_H1(9))+ min_Hout(row_locmin_H2(row_locmin_H1(9)));
NeuronsMat_Feature_9_min = [9              row_locmin_H1(9)     row_locmin_H2(row_locmin_H1(9));
                       row_locmin_H1(9)    row_locmin_H2(row_locmin_H1(9))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(9)))
];

%%%%Feature Path (10)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 10 (Max)
Input10_pathVal_max_Layer1 = max_H1(10)+ max_H2(row_locmax_H1(10))+ max_Hout(row_locmax_H2(row_locmax_H1(10)));
NeuronsMat_Feature_10_Max = [10              row_locmax_H1(10)     row_locmax_H2(row_locmax_H1(10));
                       row_locmax_H1(10)    row_locmax_H2(row_locmax_H1(10))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(10)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 10 (Min)
Input10_pathVal_min_Layer1 = min_H1(10)+ min_H2(row_locmin_H1(10))+ min_Hout(row_locmin_H2(row_locmin_H1(10)));
NeuronsMat_Feature_10_min = [10              row_locmin_H1(10)     row_locmin_H2(row_locmin_H1(10));
                       row_locmin_H1(10)    row_locmin_H2(row_locmin_H1(10))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(10)))
];

%%%%Feature Path (11)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 11 (Max)
Input11_pathVal_max_Layer1 = max_H1(11)+ max_H2(row_locmax_H1(11))+ max_Hout(row_locmax_H2(row_locmax_H1(11)));
NeuronsMat_Feature_11_Max = [11              row_locmax_H1(11)     row_locmax_H2(row_locmax_H1(11));
                       row_locmax_H1(11)    row_locmax_H2(row_locmax_H1(11))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(11)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 11 (Min)
Input11_pathVal_min_Layer1 = min_H1(11)+ min_H2(row_locmin_H1(11))+ min_Hout(row_locmin_H2(row_locmin_H1(11)));
NeuronsMat_Feature_11_min = [11              row_locmin_H1(11)     row_locmin_H2(row_locmin_H1(11));
                       row_locmin_H1(11)    row_locmin_H2(row_locmin_H1(11))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(11)))
];

%%%%Feature Path (12)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Path for Feature 12 (Max)
Input12_pathVal_max_Layer1 = max_H1(12)+ max_H2(row_locmax_H1(12))+ max_Hout(row_locmax_H2(row_locmax_H1(12)));
NeuronsMat_Feature_12_Max = [12              row_locmax_H1(12)     row_locmax_H2(row_locmax_H1(12));
                       row_locmax_H1(12)    row_locmax_H2(row_locmax_H1(12))     row_locmax_Hout(row_locmax_H2(row_locmax_H1(12)))
]; % [Input, Output] -> Weight matrix: All are of this form
%Path for Feature 12 (Min)
Input12_pathVal_min_Layer1 = min_H1(12)+ min_H2(row_locmin_H1(12))+ min_Hout(row_locmin_H2(row_locmin_H1(12)));
NeuronsMat_Feature_12_min = [12              row_locmin_H1(12)     row_locmin_H2(row_locmin_H1(12));
                       row_locmin_H1(12)    row_locmin_H2(row_locmin_H1(12))     row_locmin_Hout(row_locmin_H2(row_locmin_H1(12)))
];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Feature_Impact_max = [Input1_pathVal_max_Layer1 Input2_pathVal_max_Layer1 ...
    Input3_pathVal_max_Layer1 Input4_pathVal_max_Layer1 Input5_pathVal_max_Layer1...
    Input6_pathVal_max_Layer1 Input7_pathVal_max_Layer1 Input8_pathVal_max_Layer1...
    Input9_pathVal_max_Layer1 Input10_pathVal_max_Layer1 Input11_pathVal_max_Layer1 ...
    Input12_pathVal_max_Layer1 
];

Feature_Impact_min = [Input1_pathVal_min_Layer1 Input2_pathVal_min_Layer1 ...
    Input3_pathVal_min_Layer1 Input4_pathVal_min_Layer1 Input5_pathVal_min_Layer1...
    Input6_pathVal_min_Layer1 Input7_pathVal_min_Layer1 Input8_pathVal_min_Layer1...
    Input9_pathVal_min_Layer1 Input10_pathVal_min_Layer1 Input11_pathVal_min_Layer1 ...
    Input12_pathVal_min_Layer1 
];

feat_num = 1:12;
Feature_Count = categorical({'Heat Rate 1','Ramp 1 Duration','Temperature Dwell 1','Heat Rate 2',...
'Ramp 2 Duration','Temperature Dwell 2','Vacuum Pressure','Vacuum Start Time','Vacuum Duration', ...
'Autoclave Pressure','Autoclave Start Time','Autoclave Duration'});
Output_count = categorical({'AD Porosity','PR_Porosity','Eff_Porosity',...
'Max_Fiber_vol_fraction','Cure_Cycle_Total_Time','AD_Volume','PR_Volume';}); %Keep if we want to use with plot later
mean_Feature_impact = (Feature_Impact_max+ Feature_Impact_min)/2;

%Error bar- type chart to show overall role of parameters
figure(2)
errorbar(feat_num, mean_Feature_impact,Feature_Impact_min,Feature_Impact_max,'o','MarkerEdgeColor','red','MarkerFaceColor','red','LineWidth',1.4)
xlim([0 13])
xlabel('Input','Fontsize',12)
ylabel('Acumalitive NN path weight','Fontsize',12) 
title('Comparison of Input weights')

%Map the inputs to the outputs for highest impact (Simple bar chart)
In1_outmap = NeuronsMat_Feature_1_Max(2,3);
In2_outmap = NeuronsMat_Feature_2_Max(2,3);
In3_outmap = NeuronsMat_Feature_3_Max(2,3);
In4_outmap = NeuronsMat_Feature_4_Max(2,3);
In5_outmap = NeuronsMat_Feature_5_Max(2,3);
In6_outmap = NeuronsMat_Feature_6_Max(2,3);
In7_outmap = NeuronsMat_Feature_7_Max(2,3);
In8_outmap = NeuronsMat_Feature_8_Max(2,3);
In9_outmap = NeuronsMat_Feature_9_Max(2,3);
In10_outmap = NeuronsMat_Feature_10_Max(2,3);
In11_outmap = NeuronsMat_Feature_11_Max(2,3);
In12_outmap = NeuronsMat_Feature_12_Max(2,3);
Outmap_max = [In1_outmap,In2_outmap,In3_outmap,In4_outmap,In5_outmap, ...
          In6_outmap,In7_outmap,In8_outmap,In9_outmap,In10_outmap, ...
          In11_outmap,In12_outmap];
 %Map inputs to least effected output     
In11_outmap = NeuronsMat_Feature_1_min(2,3);
In21_outmap = NeuronsMat_Feature_2_min(2,3);
In31_outmap = NeuronsMat_Feature_3_min(2,3);
In41_outmap = NeuronsMat_Feature_4_min(2,3);
In51_outmap = NeuronsMat_Feature_5_min(2,3);
In61_outmap = NeuronsMat_Feature_6_min(2,3);
In71_outmap = NeuronsMat_Feature_7_min(2,3);
In81_outmap = NeuronsMat_Feature_8_min(2,3);
In91_outmap = NeuronsMat_Feature_9_min(2,3);
In101_outmap = NeuronsMat_Feature_10_min(2,3);
In111_outmap = NeuronsMat_Feature_11_min(2,3);
In121_outmap = NeuronsMat_Feature_12_min(2,3);
Outmap_min = [In11_outmap,In21_outmap,In31_outmap,In41_outmap,In51_outmap, ...
          In61_outmap,In71_outmap,In81_outmap,In91_outmap,In101_outmap, ...
          In111_outmap,In121_outmap];
Outmap = [Outmap_max; Outmap_min];

figure(3)
bar(Feature_Count,Outmap,'LineWidth',1.25)
ylim([0 7.5])
ylabel('Outputs (By Class)','Fontsize',12) 
title('Input-Output mapping')

%%%Prepare Surfaces of the weights for figure on methods
figure(4)
surf(Weight_hidden1)
title('Hidden Layer 1 Weight mapping')
xlabel('Input','Fontsize',12)
ylabel('Output','Fontsize',12)
zlabel('Weight','Fontsize',12)
colorbar

figure(5)
surf(Weight_hidden2)
title('Hidden Layer 2 Weight mapping')
xlabel('Input','Fontsize',12)
ylabel('Output','Fontsize',12)
zlabel('Weight','Fontsize',12)
colorbar

figure(6)
surf(Weight_toOut)
title('Final Layer Weight mapping')
xlabel('Input','Fontsize',12)
ylabel('Output','Fontsize',12)
zlabel('Weight','Fontsize',12)
colorbar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Optimization Section
%Base on input x, using trained weights and biases calc. output weights to
%correlate system output. y_1 and y_2, and obj prior to cost function modification, are listed in the following two lines to show cost function
%internal mechanisms
%y_1 = Weight_hidden1 * x_vec + Bias_hidden1;
%y_2 = Weight_hidden2 * y_1 + Bias_hidden2;
%Obj = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * xvec + Bias_hidden1) + Bias_hidden2)) + Bias_toOut;

%Inputs spanning input range for optimzation
x1 = 0.5:0.35:4;   
x2 = 20:10:120;  
x3 = 40:10:140;
x4 = 0:0.5:5;  
x5 = 10:9:100; 
x6 = 0:16:160; 
x7 = 0:0.1:1; 
x8 = 0:10:100; 
x9 = 50:30:350;  
x10 = 1:0.5:6;
x11 = 1:9.9:100;
x12 = 100:25:350;

%Due to the very large computation space the location with the optimal
%repsonse is taken for each var individually
%This next section is not elegant,because the loop was not iterating on x_k(i), so 100
%lines are hard coded.
yout11 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(1); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout12 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(2); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout13 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(3); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout14 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(4); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout15 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(5); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout16 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(6); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout17 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(7); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout18 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(8); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout19 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(9); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout110 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(10); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout1 = [yout11 yout12 yout13 yout14 yout15 yout16 yout17 yout18 yout19 yout110 ];
[~, yout1Pos] = min(yout1'); %To explain yout1 are the output values I want the position  of the min. then i search for the highest correlated one
yp1 = max(yout1Pos);
%%%%
yout21 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(1); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout22 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(2); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout23 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(3); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout24 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(4); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout25 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(5); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout26 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(6); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout27 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(7); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout28 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(8); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout29 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(9); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout210 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(10); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout2 = [yout21 yout22 yout23 yout24 yout25 yout26 yout27 yout28 yout29 yout210 ];
[~, yout2Pos] = min(yout2');
yp2 = max(yout2Pos);
%%%%
yout31 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(1); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout32 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(2); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout33 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(3); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout34 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(4); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout35 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(5); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout36 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(6); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout37 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(7); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout38 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(8); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout39 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(9); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout310 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(10); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout3 = [yout31 yout32 yout33 yout34 yout35 yout36 yout37 yout38 yout39 yout310 ];
[~, yout3Pos] = min(yout3');
yp3 = max(yout3Pos);
%%%%
yout41 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(1); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout42 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(2); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout43 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(3); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout44 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(4); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout45 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(5); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout46 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(6); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout47 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(7); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout48 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(8); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout49 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(9); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout410 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(10); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout4 = [yout41 yout42 yout43 yout44 yout45 yout46 yout47 yout48 yout49 yout410 ];
[~, yout4Pos] = min(yout4');
yp4 = max(yout4Pos);
%%%%
yout51 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(1); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout52 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(2); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout53 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(3); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout54 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(4); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout55 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(5); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout56 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(6); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout57 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(7); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout58 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(8); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout59 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(9); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout510 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(10); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout5 = [yout51 yout52 yout53 yout54 yout55 yout56 yout57 yout58 yout59 yout510 ];
[~, yout5Pos] = min(yout5');
yp5 = max(yout5Pos);
%%%%
yout61 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(1); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout62 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(2); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout63 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(3); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout64 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(4); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout65 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(5); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout66 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(6); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout67 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(7); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout68 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(8); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout69 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(9); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout610 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(10); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout6 = [yout61 yout62 yout63 yout64 yout65 yout66 yout67 yout68 yout69 yout610 ];
[~, yout6Pos] = min(yout6');
yp6 = max(yout6Pos);
%%%%
yout71 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(1); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout72 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(2); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout73 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(3); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout74 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(4); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout75 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(5); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout76 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(6); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout77 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(7); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout78 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(8); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout79 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(9); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout710 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(10); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout7 = [yout71 yout72 yout73 yout74 yout75 yout76 yout77 yout78 yout79 yout710 ];
[~, yout7Pos] = min(yout7');
yp7 = max(yout7Pos);
%%%%
yout81 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(1); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout82 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(2); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout83 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(3); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout84 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(4); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout85 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(5); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout86 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(6); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout87 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(7); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout88 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(8); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout89 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(9); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout810 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(10); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout8 = [yout81 yout82 yout83 yout84 yout85 yout86 yout87 yout88 yout89 yout810 ];
[~, yout8Pos] = min(yout8');
yp8 = max(yout8Pos);
%%%%
yout91 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(1); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout92 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(2); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout93 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(3); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout94 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(4); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout95 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(5); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout96 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(6); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout97 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(7); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout98 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(8); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout99 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(9); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout910 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(10); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout9 = [yout91 yout92 yout93 yout94 yout95 yout96 yout97 yout98 yout99 yout910 ];
[~, yout9Pos] = min(yout9');
yp9 = max(yout9Pos);
%%%%
yout101 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(1); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout102 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(2); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout103 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(3); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout104 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(4); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout105 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(5); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout106 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(6); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout107 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(7); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout108 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(8); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout109 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(9); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout1010 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(10); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout10 = [yout101 yout102 yout103 yout104 yout105 yout106 yout107 yout108 yout109 yout1010 ];
[~, yout10Pos] = min(yout10');
yp10 = max(yout10Pos);
%%%%
yout111 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(1); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout112 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(2); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout113 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(3); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout114 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(4); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout115 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(5); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout116 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(6); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout117 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(7); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout118 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(8); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout119 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(9); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout1110 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(10); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout11 = [yout111 yout112 yout113 yout114 yout115 yout116 yout117 yout118 yout119 yout1110 ];
[~, yout11Pos] = min(yout11');
yp11 = max(yout11Pos);
%%%%
yout121 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(1)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout122 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(2)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout123 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(3)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout124 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(4)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout125 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(5)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout126 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(6)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout127 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(7)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout128 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(8)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout129 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(9)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout1210 = (Weight_toOut * (Weight_hidden2 * (Weight_hidden1 * [x1(yp1); x2(yp2); x3(yp3); x4(yp4); x5(yp5); x6(yp6); x7(yp7); x8(yp8); x9(yp9); x10(yp10); x11(yp11); x12(10)] + Bias_hidden1) + Bias_hidden2))+ Bias_toOut;
yout12 = [yout121 yout122 yout123 yout124 yout125 yout126 yout127 yout128 yout129 yout1210 ];
[~, yout12Pos] = min(yout12');
yp12 = max(yout12Pos);

%Get Optimal Solution
Optimal = [x1(yp1) x2(yp2) x3(yp3) x4(yp4) x5(yp5) x6(yp6) x7(yp7) x8(yp8) x9(yp9) x10(yp10) x11(yp11) x12(yp12)];
disp(Optimal);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Export essential data [Optional]
csvwrite('XTrain',XTrain);
csvwrite('YTrain',YTrain);
csvwrite('XValidation',XValidation);
csvwrite('YValidation',YValidation);
csvwrite('YPred',YPred);
csvwrite('rmse',rmse);
csvwrite('Weight_hidden1',Weight_hidden1);
csvwrite('Weight_hidden2',Weight_hidden2);
csvwrite('Weight_toOut',Weight_toOut);
csvwrite('Bias_hidden1',Bias_hidden1);
csvwrite('Bias_hidden2',Bias_hidden2);
csvwrite('Bias_toOut',Bias_toOut);
csvwrite('Optimal',Optimal);
