# EIP-4


All the contents are assignment for EIP 4 course

# Assignment 2
The first task of the assignment done is to reduce the number of the parameters without making any changes such as using fully connected layer (FCL) or bias. So, I tried to tune the rest of the paramters present in the preexisting code. I removed the batchnormalization and dropuot from the last convolutional layer as dirested by Rohan sir. First, I tried to implement the MaxPooling of 2x2 lyaer after after 9x9 and before 7x7 output dimention. But it didn't help as it was successful in decreasing the number of the parameter to 11k but at the time of training it was having score of 99.2 which as less than 0.2 from the target score of 99.4. I also tried to decrease some of the convolutional 2d layer but it didn't help. Then I tried to increase the number of filter such as 16, 32, 10, 16, 32, 64, 128, 10. The number of paramter incresed to some 1L. I also tried to increase the number of channels in 1x1 convolutional layer but the score was not able to reach 99.4 though it was having 15k parameters. Then I reset to its original value except two. The second convolutional layer channel is decresed from 32 to 16 making same for the rest of the convolutional 2d layer (not 1x1 convolutional layer). The dropout value is incresed from 0.1 to 0.2. This dectreasssed the number of the parametr to 14k which is near to 15k. So, Trained the model to chek the accuracy. The score is 99.39 whis is near to 99.4 for 20 epochs. 

# Assignment 3
