# projectone

This is project one of the A.I course. The problem is given two numbers from 1 to 100 predict the addition of these two number. 
So for example 15+16 = 31 the model should be able to predict 31. The normal logistic regression should be able to solve this problem but
as the instructor want us to practice on the basic of deep learning network , this problem was thus set for us. 

Initially one set of all combination of addition from 1 to 100 was not enough for the model to learn and generalise and thus the data set 
was increased to 100 sets of all combination.  80 % of this data was then used for test and training purposes and 20 % for evaluation later. 

A six layer model was choosen with random units from 250 to 50 all with relu activation except with the last layer using softmax. 

A accuracy of 95.41 percent was achieved for train data and validation accuracy was at 95.57. Evaluation accuracy was at 94.66 percent.
