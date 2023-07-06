# **CS50AI Week 5 Project: Traffic**

## Objective: Write an AI to identify which traffic sign appears in a photograph

## Experimentation Process
I wrote a function which initialized and compiled various Convolutional Neural Network Models.
The models had either
1. 1 or 2 convolutional and pooling layers
1. 32 or 64 3x3 filters
1. 2x2 or 4x4 pooling sizes
1. Hidden layer of 256 or 512 units
1. Dropout of 50% or no dropout 

Each model was trained and evaluated 10 times, and their average performance was recorded in a dataframe which was then transferred to an excel sheet.

The top 5 model configurations in terms of test accuracy is as follows:

|                                                          | Training Loss | Test Loss   | Training Accuracy | Test Accuracy |
| -------------------------------------------------------- | ------------- | ----------- | ----------------- | ------------- |
| two_conv_64_3x3_filter_2x2_pooling_512_hidden_w_dropout  | 0.18116294    | 0.087751367 | 0.97179054        | 0.984647149   |
| two_conv_32_3x3_filter_2x2_pooling_512_hidden_w_dropout  | 0.179525579   | 0.094031106 | 0.973948956       | 0.984159154   |
| two_conv_64_3x3_filter_2x2_pooling_256_hidden_wo_dropout | 0.067285594   | 0.207257193 | 0.991241246       | 0.983220726   |
| two_conv_32_3x3_filter_2x2_pooling_256_hidden_w_dropout  | 0.164723649   | 0.076102293 | 0.966622865       | 0.983136261   |
| two_conv_32_3x3_filter_2x2_pooling_0_hidden_wo_dropout   | 0.046324108   | 0.153860315 | 0.992880374       | 0.982732743   |

Looking at the training and test accuracies, I found the best configuration to be:
1. 2 convolutional and pooling layers
1. 64 3x3 filters
1. 2x2 pooling size
1. 256 units in hidden layer
1. No dropout

This configuration has a high training accuracy as well as test accuracy. Compared to the model with the highest test accuracy, this configuration only has a slightly lower test accuracy, but significantly higher training accuracy. This high training accuracy could be due to the lack of dropout, but it seems to be able to generalize well to unseen data (at least for this specific dataset)

Additional Findings:
Effect of pooling size: Out of the 40 models evaluated (20 with 2x2 pooling size, 20 with 4x4 pooling size), models with 4x4 pooling size tend to perform poorer than their 2x2 counterparts. 18 out of 20 of the worst performing models in terms of test accuracy were models with 4x4 pooling size.








