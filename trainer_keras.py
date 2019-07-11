from keras.layers import Lambda, Input
from keras import optimizers
from IDG import DataGenerator
from loss_fn import euclidean_distance, eucl_dist_output_shape, contrastive_loss, accuracy
from base_network import get_base_net

input_a = Input(shape=(155,220,3))
input_b = Input(shape=(155,220,3))

base_net = get_base_net()
processed_a = base_net(input_a)
processed_b = base_net(input_b)

distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)


optimizer = optimizers.RMSprop()
model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])

params={
    'dim': (155,220),
    'batch_size': 32,
    'n_channels': 3,
    'shuffle': False
}

#train_set = 2592*0.7 = 1814 == 1814/32 == 56
#val_set = 2592*0.3 = 777 == 777/32 == 24
ds_train = get_dataset("train")
ds_val = get_dataset("val")
train_datagen = DataGenerator(ds_train,**params)
validation_datagen = DataGenerator(ds_val,**params)
model.fit_generator(generator=train_datagen, validation_data=validation_datagen, epochs=30, steps_per_epoch=56, validation_steps=24, use_multiprocessing=True, workers=6)
model.save("drive/My Drive/Colab Notebooks/SigNet/colab_08072019.h5")