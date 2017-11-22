import keras
import math

from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler


num = 100
train_epochs = 500
batch_sizes = 10000

x_list = []
y_list = []

for mul in range(0,100):
    for i in range(1,num+1):
        for r in range(1,num+1):
            x_list.append([i,r])
            y_list.append(i+r)





n_classes = 201


np.random.seed(seed=315)

percentage = 0.8

train_x  = np.array(x_list).astype('float32')
train_y  = np.array(y_list).astype('float32')

n_train = len(train_y)


training_size = int(percentage*n_train)
mask=np.random.permutation(np.arange(n_train))[:training_size]
x_train, y_train = train_x[mask], train_y[mask]

y_train = to_categorical(y_train,n_classes)

x_train = x_train / 100




n_input = 2
n_hidden_1 = 150
n_hidden_2 = 200
n_hidden_3 = 100
n_hidden_4 = 250
n_hidden_5 = 100
n_hidden_6 = 50

Inp = Input(shape=(2,))
x = Dense( n_hidden_1, activation = 'relu', name = 'Dense_1')(Inp)
x = Dense( n_hidden_2, activation = 'relu', name = 'Dense_2')(x)
x = Dense( n_hidden_3, activation = 'relu', name = 'Dense_3')(x)
x = Dense( n_hidden_4, activation = 'relu', name = 'Dense_4')(x)
x = Dense( n_hidden_5, activation = 'relu', name = 'Dense_5')(x)

x = Dense( n_hidden_6, activation = 'relu', name = 'Dense_6')(x)

output = Dense( n_classes, activation = 'softmax', name = 'Dense_out')(x)


model = Model(Inp, output)


def step_decay(epoch):

    if epoch<20:
        return 0.1
    elif epoch>60:
        return 0.01
    else:
        return 0.001

    return 0.0005



lrate=LearningRateScheduler(step_decay)



model.get_weights()
adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)
model.compile( loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'] )



history = model.fit(x_train, y_train,
                    batch_size = batch_sizes,
                    epochs = train_epochs, verbose = 1,
                    validation_split=0.2,callbacks=[lrate],shuffle=True )

model.save_weights('addition_new2.h5')


model.save('mymodel.h5')

