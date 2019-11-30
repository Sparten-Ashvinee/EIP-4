# EIP-4


All the contents are assignment for EIP 4 course

# Assignment 2
The first task of the assignment done is to reduce the number of the parameters without making any changes such as using fully connected layer (FCL) or bias. So, I tried to tune the rest of the paramters present in the preexisting code. I removed the batchnormalization and dropuot from the last convolutional layer as dirested by Rohan sir. First, I tried to implement the MaxPooling of 2x2 lyaer after after 9x9 and before 7x7 output dimention. But it didn't help as it was successful in decreasing the number of the parameter to 11k but at the time of training it was having score of 99.2 which as less than 0.2 from the target score of 99.4. I also tried to decrease some of the convolutional 2d layer but it didn't help. Then I tried to increase the number of filter such as 16, 32, 10, 16, 32, 64, 128, 10. The number of paramter incresed to some 1L. I also tried to increase the number of channels in 1x1 convolutional layer but the score was not able to reach 99.4 though it was having 15k parameters. Then I reset to its original value except two. The second convolutional layer channel is decresed from 32 to 16 making same for the rest of the convolutional 2d layer (not 1x1 convolutional layer). The dropout value is incresed from 0.1 to 0.2. This dectreasssed the number of the parametr to 14k which is near to 15k. So, Trained the model to chek the accuracy. The score is 99.39 whis is near to 99.4 for 20 epochs. 

# Assignment 3

Final validation accuracy = 82.61

Model Definitation:
model = Sequential()
model.add(SeparableConv2D(48, (3, 3), border_mode='same', input_shape=(32, 32, 3)))   #32x32x48   #3
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(SeparableConv2D(48, (3, 3),))   #30x30x48   #5
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(128, (1, 1)))   #30x30x128   #5
model.add(MaxPooling2D(pool_size=(2, 2)))   #15x15x128   #6
model.add(BatchNormalization())
model.add(SeparableConv2D(96, (3, 3), border_mode='same'))    #15x15x96   #10
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(SeparableConv2D(96, (3, 3)))    #13x13x96   #14
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(200, (1, 1)))   #13x13x200   #14
model.add(MaxPooling2D(pool_size=(2, 2)))   #6x6x200    #16
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(SeparableConv2D(192, (3, 3), border_mode='same'))   #6x6x192    #24
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(SeparableConv2D(10, (4, 4)))    #3x3x10    #36
model.add(Activation('relu'))
model.add(Dropout(0.1))
#model.add(Convolution2D(128, (1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))   #1x1x10    #40
model.add(BatchNormalization())
#model.add(Dropout(0.25))
model.add(Flatten())   
#model.add(Dense(num_classes, activation='softmax'))
#model.add(Convolution2D(10, (1, 1)))
model.add(Activation('softmax'))
# Compile the model
adam = Adam(lr=0.08, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

50 epoch logs:
Epoch 1/50
390/390 [==============================] - 48s 124ms/step - loss: 1.7711 - acc: 0.3551 - val_loss: 1.5053 - val_acc: 0.4653
Epoch 2/50
390/390 [==============================] - 42s 109ms/step - loss: 1.4625 - acc: 0.4843 - val_loss: 1.4612 - val_acc: 0.4752
Epoch 3/50
390/390 [==============================] - 42s 109ms/step - loss: 1.3324 - acc: 0.5324 - val_loss: 1.2072 - val_acc: 0.5795
Epoch 4/50
390/390 [==============================] - 42s 109ms/step - loss: 1.2422 - acc: 0.5670 - val_loss: 1.1754 - val_acc: 0.5848
Epoch 5/50
390/390 [==============================] - 42s 108ms/step - loss: 1.1719 - acc: 0.5923 - val_loss: 1.0197 - val_acc: 0.6459
Epoch 6/50
390/390 [==============================] - 42s 109ms/step - loss: 1.1139 - acc: 0.6146 - val_loss: 0.9653 - val_acc: 0.6693
Epoch 7/50
390/390 [==============================] - 42s 108ms/step - loss: 1.0629 - acc: 0.6317 - val_loss: 0.9546 - val_acc: 0.6746
Epoch 8/50
390/390 [==============================] - 42s 108ms/step - loss: 1.0286 - acc: 0.6430 - val_loss: 0.8314 - val_acc: 0.7230
Epoch 9/50
390/390 [==============================] - 42s 109ms/step - loss: 0.9912 - acc: 0.6568 - val_loss: 0.8438 - val_acc: 0.7088
Epoch 10/50
390/390 [==============================] - 42s 108ms/step - loss: 0.9640 - acc: 0.6658 - val_loss: 0.8207 - val_acc: 0.7219
Epoch 11/50
390/390 [==============================] - 42s 108ms/step - loss: 0.9470 - acc: 0.6730 - val_loss: 0.8262 - val_acc: 0.7105
Epoch 12/50
390/390 [==============================] - 42s 109ms/step - loss: 0.9167 - acc: 0.6826 - val_loss: 0.8232 - val_acc: 0.7186
Epoch 13/50
390/390 [==============================] - 42s 109ms/step - loss: 0.8969 - acc: 0.6888 - val_loss: 0.7126 - val_acc: 0.7581
Epoch 14/50
390/390 [==============================] - 42s 109ms/step - loss: 0.8762 - acc: 0.6960 - val_loss: 0.8067 - val_acc: 0.7223
Epoch 15/50
390/390 [==============================] - 42s 109ms/step - loss: 0.8651 - acc: 0.7011 - val_loss: 0.7683 - val_acc: 0.7321
Epoch 16/50
390/390 [==============================] - 42s 109ms/step - loss: 0.8509 - acc: 0.7064 - val_loss: 0.7880 - val_acc: 0.7293
Epoch 17/50
390/390 [==============================] - 42s 108ms/step - loss: 0.8363 - acc: 0.7106 - val_loss: 0.6841 - val_acc: 0.7713
Epoch 18/50
390/390 [==============================] - 42s 108ms/step - loss: 0.8232 - acc: 0.7154 - val_loss: 0.7101 - val_acc: 0.7535
Epoch 19/50
390/390 [==============================] - 42s 108ms/step - loss: 0.8190 - acc: 0.7147 - val_loss: 0.7378 - val_acc: 0.7503
Epoch 20/50
390/390 [==============================] - 42s 109ms/step - loss: 0.8009 - acc: 0.7234 - val_loss: 0.6934 - val_acc: 0.7598
Epoch 21/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7913 - acc: 0.7282 - val_loss: 0.6621 - val_acc: 0.7743
Epoch 22/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7768 - acc: 0.7315 - val_loss: 0.7372 - val_acc: 0.7474
Epoch 23/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7781 - acc: 0.7287 - val_loss: 0.6578 - val_acc: 0.7731
Epoch 24/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7648 - acc: 0.7352 - val_loss: 0.6243 - val_acc: 0.7830
Epoch 25/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7539 - acc: 0.7381 - val_loss: 0.6454 - val_acc: 0.7785
Epoch 26/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7479 - acc: 0.7409 - val_loss: 0.6401 - val_acc: 0.7800
Epoch 27/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7408 - acc: 0.7430 - val_loss: 0.6672 - val_acc: 0.7720
Epoch 28/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7313 - acc: 0.7476 - val_loss: 0.6453 - val_acc: 0.7781
Epoch 29/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7277 - acc: 0.7468 - val_loss: 0.6753 - val_acc: 0.7681
Epoch 30/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7290 - acc: 0.7480 - val_loss: 0.6400 - val_acc: 0.7809
Epoch 31/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7161 - acc: 0.7504 - val_loss: 0.6369 - val_acc: 0.7781
Epoch 32/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7100 - acc: 0.7538 - val_loss: 0.6088 - val_acc: 0.7947
Epoch 33/50
390/390 [==============================] - 42s 108ms/step - loss: 0.7074 - acc: 0.7560 - val_loss: 0.6024 - val_acc: 0.7935
Epoch 34/50
390/390 [==============================] - 42s 109ms/step - loss: 0.7003 - acc: 0.7565 - val_loss: 0.5736 - val_acc: 0.8097
Epoch 35/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6954 - acc: 0.7610 - val_loss: 0.5973 - val_acc: 0.7965
Epoch 36/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6903 - acc: 0.7606 - val_loss: 0.5692 - val_acc: 0.8079
Epoch 37/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6838 - acc: 0.7633 - val_loss: 0.5617 - val_acc: 0.8091
Epoch 38/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6827 - acc: 0.7650 - val_loss: 0.6137 - val_acc: 0.7912
Epoch 39/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6756 - acc: 0.7676 - val_loss: 0.5589 - val_acc: 0.8088
Epoch 40/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6769 - acc: 0.7665 - val_loss: 0.5397 - val_acc: 0.8163
Epoch 41/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6685 - acc: 0.7676 - val_loss: 0.5276 - val_acc: 0.8161
Epoch 42/50
390/390 [==============================] - 42s 109ms/step - loss: 0.6657 - acc: 0.7684 - val_loss: 0.5812 - val_acc: 0.8010
Epoch 43/50
390/390 [==============================] - 42s 109ms/step - loss: 0.6532 - acc: 0.7740 - val_loss: 0.5322 - val_acc: 0.8191
Epoch 44/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6590 - acc: 0.7718 - val_loss: 0.5453 - val_acc: 0.8158
Epoch 45/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6469 - acc: 0.7764 - val_loss: 0.5301 - val_acc: 0.8165
Epoch 46/50
390/390 [==============================] - 42s 108ms/step - loss: 0.6495 - acc: 0.7747 - val_loss: 0.5944 - val_acc: 0.7952
Epoch 47/50
390/390 [==============================] - 43s 109ms/step - loss: 0.6498 - acc: 0.7760 - val_loss: 0.5273 - val_acc: 0.8189
Epoch 48/50
390/390 [==============================] - 43s 109ms/step - loss: 0.6358 - acc: 0.7815 - val_loss: 0.5864 - val_acc: 0.8013
Epoch 49/50
390/390 [==============================] - 42s 109ms/step - loss: 0.6390 - acc: 0.7778 - val_loss: 0.6156 - val_acc: 0.7954
Epoch 50/50
390/390 [==============================] - 42s 109ms/step - loss: 0.6424 - acc: 0.7785 - val_loss: 0.5114 - val_acc: 0.8261

