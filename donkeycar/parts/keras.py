""""

keras.py

functions to run and train autopilots using keras

"""

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D, BatchNormalization, Concatenate, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Cropping2D, Lambda
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from donkeycar import util


class KerasPilot:

    def load(self, model_path):
        self.model = load_model(model_path)

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)
        return hist


class KerasCategorical(KerasPilot):
    def __init__(self, model=None, *args, **kwargs):
        super(KerasCategorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = default_categorical()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        angle_unbinned = util.data.linear_unbin(angle_binned[0])
        return angle_unbinned, throttle[0][0]


class KerasLinear(KerasPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        print("DASDSASDSASAD")
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = default_n_linear(num_outputs)
        else:
            print("DASDSASDSASAD")
            self.model = andreas_linear()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def default_categorical():
    img_in = Input(shape=(120, 160, 3),
                   name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)  # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(
        x)  # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    # continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .01})

    return model


from tensorflow.python.keras import backend as K


def linear_unbin_layer(tnsr):
    bin = K.constant((2 / 14), dtype='float32')
    norm = K.constant(1, dtype='float32')

    b = K.cast(K.argmax(tnsr), dtype='float32')
    a = b - norm
    # print('linear_unbin_layer out: {}'.format(a))
    return a


def default_catlin():
    """
    Categorial Steering output before linear conversion.
    :return:
    """
    img_in = Input(shape=(120, 160, 3),
                   name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(
        x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)  # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_cat_out = Dense(15, activation='softmax', name='angle_cat_out')(x)
    angle_out = Dense(1, activation='sigmoid', name='angle_out')(angle_cat_out)
    # angle_out = Lambda(linear_unbin_layer, output_shape=(1,1, ), name='angle_out')(angle_cat_out)

    # continuous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .01})

    return model

# defines the network architecture
def andreas_linear():
    """ returns model - see README for details """
    input_img = Input(shape=(120, 160, 3), name='img_in')
    
    # use the NORMALIZED input to initialize the 'network' variable:
    #network = BatchNormalization(epsilon=0.001, mode=1)(input_img)
    network = BatchNormalization(epsilon=0.001)(input_img)
    
    # Layer 1: 1x1 convolution with 3 outputs to decide on color channel
    network = Convolution2D(3, (1, 1), activation="relu", padding="same", kernel_regularizer="l2")(network)

    # Layer 2: Inception with 3 convolutional sub-layers: 
    ### sub-layer 1: 1x1 convolution with 16 outputs, ReLu activation and dropout afterwards
    conv1 = Convolution2D(16, (1, 1), activation="relu", padding="same", kernel_regularizer="l2")(network)
    conv1 = Dropout(0.6)(conv1)
    ### sub-layer 2: stacked two 3x3 convolutions with 16 outputs and ReLu activations and dropout afterwards
    conv3 = Convolution2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
    conv3 = Convolution2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(conv3)
    conv3 = Dropout(0.6)(conv3)
    ### sub-layer 3: stacked two 5x5 convolutions with 16 outputs and ReLu activations and dropout afterwards
    conv5 = Convolution2D(16, (5, 5), activation="relu", padding="same", kernel_regularizer="l2")(network)
    conv5 = Convolution2D(16, (5, 5), activation="relu", padding="same", kernel_regularizer="l2")(conv5)
    conv5 = Dropout(0.6)(conv5)
    ### combine results of 1x1, 3x3, and 5x5 convolutions
    #network = merge([conv1, conv3, conv5], mode='concat', concat_axis=1)
    network = Concatenate()([conv1, conv3, conv5])
    
    # Layer 3: Max-pooling:
    network = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network)
    
    # Layer 4: stacked three 3x3 convolutions with 32 outputs and ReLu activations and dropout afterwards
    network = Convolution2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
    network = Convolution2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
    network = Convolution2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
    network = Dropout(0.6)(network)

    # Layer 5: Max-pooling:
    network = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network)
    
    # Layer 6: Flatten and Fully Connected layer with 128 size, ReLu activation and dropout afterwards
    network = Flatten()(network)
    network = Dense(128, activation="relu", kernel_regularizer="l2")(network)
    network = Dropout(0.6)(network)
    
    # output 'angle_out' : Layer 7: Fully Connected. linearly output to one angle value:
    angle_out = Dense(1, activation='linear', name='angle_out', kernel_regularizer="l2")(network)

    # output 'angle_out' : Layer 7: Fully Connected. linearly output to one angle value:
    throttle_out = Dense(1, activation='linear', name='throttle_out', kernel_regularizer="l2")(network)

    # return full model from input to outputs:
    model = Model(input_img, outputs=[angle_out, throttle_out])
 
    model.compile(optimizer='adam',
                         loss={'angle_out': 'mse', 'throttle_out': 'mse'},
                         loss_weights={'angle_out': .98, 'throttle_out': .02}) # always full-throttle seems to be fine for now
    return model

def default_linear():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='linear')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='linear')(x)
    x = Dropout(.1)(x)
    # categorical output of the angle
    angle_out = Dense(1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model


def default_n_linear(num_outputs):
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    x = Cropping2D(cropping=((60, 0), (0, 0)))(x)  # trim 60 pixels off top
    x = Lambda(lambda x: x / 127.5 - 1.)(x)  # normalize and re-center
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(64, (5, 5), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)

    outputs = []

    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs)

    model.compile(optimizer='adam',
                  loss='mse')

    return model
