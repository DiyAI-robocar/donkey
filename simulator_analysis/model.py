import numpy as np
import pandas as pd
import json
import cv2
import time
import glob

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


# threshold used to ignore consecutive images (as they're basically the same scene)
img_downsample_threshold = 1

image_pos = 0
angle_pos = 1  
throttle_pos = 2  

# Fix paths on windows machines:
def fix_windows_path(path):
    return path.replace("\\", "/")

def fix_windows_paths(list_of_paths):
    for i, entry in enumerate(list_of_paths):
        list_of_paths[i] = fix_windows_path(entry)
    return list_of_paths

    
# reads in all files that match: $path/$prefix*/record_*
# and returns file location, angle and throttle from these as numpy array (as well as lengths of each folder)
def get_data(path, prefix): 
    
    content = None
    content_lengths = []
    
    # first make sure path ends with final '/'
    path = path if path.endswith('/') else (path + '/')
    glob_path = path + prefix + '*'
    print(glob_path)
    # generate list of full paths from prefix:
    path_list = fix_windows_paths(glob.glob(glob_path))
    print ("path_list: "  + str(path_list))
    # read all images into numpy array:
    for entry in path_list:
        # find all record_* json files:
        record_glob_path=entry + '/record_' + '*'
        print(record_glob_path)
        json_list = fix_windows_paths(glob.glob(record_glob_path))
        curr_content = np.empty(shape=(len(json_list),), dtype=np.dtype('<U150,f,f'))
        
        first_image_idx = None
        print (json_list)
        for i, json_file in enumerate(json_list):
            print (json_file)
            with open(json_file) as f:
                data = json.load(f)
                print(data)
                proper_i = int(data['cam/image_array'].split('_',1)[0]) # get number of image from path name
                first_image_idx = proper_i if first_image_idx is None else first_image_idx
                print("fdsssdfsdf",first_image_idx, proper_i);
                curr_content[proper_i - first_image_idx] = (entry + '/' +data['cam/image_array'], data['user/angle'], data['user/throttle'])
        
        content = curr_content if content is None else np.concatenate([content,curr_content])
        content_lengths = content_lengths + [len(json_list)]
    
    return content, content_lengths
    
# given the output of get_data() this functions:
#  - removes entries around all folder-borders (incl first and last images)
#  - smooths the angles as a rolling average with fixed window-size
def adjust_data(data, lengths):
    # moving window border
    N=1

    adjusted = data.copy()
    ends = [0] + list(np.add.accumulate(lengths))

    nan_index_list = []
    for end in ends:
        for i in range(int(N/2)):
            nan_index_list = nan_index_list + [min(max(0,end-i),max(ends)-1), min(end+i,max(ends)-1)]
    nan_index_list = sorted(set(nan_index_list))

    for idx in nan_index_list:
        adjusted[idx][angle_pos] = np.nan
        #adjusted[idx][throttle_pos] = np.nan

    angle_values = [float(row[angle_pos]) for row in adjusted[0:len(adjusted)]]
    angle_values = pd.Series(angle_values).rolling(N).mean()
     
    #throttle_values = [float(row[throttle_pos]) for row in adjusted[0:len(adjusted)]]
    #throttle_values = pd.Series(throttle_values).rolling(N).mean()
    
    for idx in range(int(N/2),len(angle_values)):
        adjusted[idx-int(N/2)][angle_pos] = angle_values[idx]
        #adjusted[idx-int(N/2)][throttle_pos] = throttle_values[idx]
        
    for idx in range(int(N/2)):
        adjusted[idx][angle_pos] = np.nan
        adjusted[-idx][angle_pos] = np.nan
        #adjusted[idx][throttle_pos] = np.nan
        #adjusted[-idx][throttle_pos] = np.nan
        
    return adjusted

# gets raw data and adjusts it to be used in ReadMe notebook
def get_raw_data(path = '../../tubp', prefix = 'prefix'):
    """ Gets raw train and validation data from directories, returns both as numpy arrays"""
    
    all_content, lengths = get_data(path, prefix)
    adjusted = adjust_data(all_content, lengths)
    return all_content, adjusted, lengths


# get the angles values from output of get_data()
def get_all_angles(data):
    """ Returns all steering angles as floats for given data, uses flipped column as well! """
    return np.asfarray([float(row[angle_pos]) for row in data]).reshape(data.shape[0],1)

# get the throttle values from output of get_data()    
def get_all_throttles(data):
    """ Returns all steering angles as floats for given data, uses flipped column as well! """
    return np.asfarray([float(row[throttle_pos]) for row in data]).reshape(data.shape[0],1)
 
 
### NOTE: this function needs to be duplicated in base.py and used on input images!!
def preprocess_image(image, horizon = 0.4, y_ratio = 0.7, x_ratio = 0.5):
    """
    Given an image as an array, this function crops the top 'horizon' percentage off it and
    resizes the image afterwards according to the given x and y ratios.
    
    :param image: given image as array
    :param horizon: top percentage to be cut off - considered to be the horizon (default: 40%)
    :param y_ratio: resize ratio in y dimension after cutting horizon: default 50%
    :param x_ratio: resize ratio in x dimension after cutting horizon: default 25%

    :return: The new image as an array
    """
    # crop out the horizon (top percentage of image):
    y_full = image.shape[0]
    image = image[int(horizon*y_full):y_full, 0:image.shape[1]]
    
    # do some resizing
    image = cv2.resize(image,(0,0), fx=x_ratio, fy=y_ratio, interpolation=cv2.INTER_LANCZOS4);
    
    return image


# For a given index in the data, return the preprocessed image as an array.
def get_image(idx, data):
    return preprocess_image(cv2.imread(data[idx][image_pos]))


## NOTE: use this as input for the model:
def get_stacked_data(data):
    """ Stacks all images of given data into a numpy array of appropriate dimensions """
    shape = get_image(0, data).shape
    stacked = np.empty((data.shape[0], shape[0], shape[1], shape[2]))
    for n in range(data.shape[0]):
        center = get_image(n, data)
        stacked[n,:,:] = center
    return stacked    



from keras.models import Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Concatenate, Input
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# defines the network architecture
def get_model(in_shape):
    """ returns model - see README for details """
    input_img = Input(shape=in_shape)
    
    # use the NORMALIZED input to initialize the 'network' variable:
    #network = BatchNormalization(epsilon=0.001, mode=1)(input_img)
    network = BatchNormalization(epsilon=0.001)(input_img)
    
    # Layer 1: 1x1 convolution with 3 outputs to decide on color channel
    network = Conv2D(3, (1, 1), activation="relu", padding="same", kernel_regularizer="l2")(network)

    # Layer 2: Inception with 3 convolutional sub-layers: 
    ### sub-layer 1: 1x1 convolution with 16 outputs, ReLu activation and dropout afterwards
    conv1 = Conv2D(16, (1, 1), activation="relu", padding="same", kernel_regularizer="l2")(network)
    conv1 = Dropout(0.6)(conv1)
    ### sub-layer 2: stacked two 3x3 convolutions with 16 outputs and ReLu activations and dropout afterwards
    conv3 = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
    conv3 = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(conv3)
    conv3 = Dropout(0.6)(conv3)
    ### sub-layer 3: stacked two 5x5 convolutions with 16 outputs and ReLu activations and dropout afterwards
    conv5 = Conv2D(16, (5, 5), activation="relu", padding="same", kernel_regularizer="l2")(network)
    conv5 = Conv2D(16, (5, 5), activation="relu", padding="same", kernel_regularizer="l2")(conv5)
    conv5 = Dropout(0.6)(conv5)
    ### combine results of 1x1, 3x3, and 5x5 convolutions
    #network = merge([conv1, conv3, conv5], mode='concat', concat_axis=1)
    network = Concatenate()([conv1, conv3, conv5])
    
    # Layer 3: Max-pooling:
    network = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(network)
    
    # Layer 4: stacked three 3x3 convolutions with 32 outputs and ReLu activations and dropout afterwards
    network = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
    network = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
    network = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer="l2")(network)
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
    return Model(input_img, outputs=[angle_out, throttle_out])

# helper to remove NaN from data received by get_data:
def not_nan_row(row, pos_idx = angle_pos):
    return ~np.isnan(row[pos_idx])

# sub-sample all data before shuffling 
def sub_sample_data(data, N=img_downsample_threshold):
    remaining_indexes = np.arange(0,len(data),N)
    return data[remaining_indexes]

    
#def train_test_split(data, train_split = 0.8):
#    split = int(train_split*len(data))
#    return data[:split], data[split:]

def train_test_splity(data, train_split = 0.8):
    return train_test_split(data, train_size=train_split, shuffle=True,random_state=42)

    
# function to get all data, split it and start training with model defined above:
def train_model(EPOCHS = 10000, verbose = True):
    """ Whole pipeline: get all data, use model to train, returns history and trained model """
    
    now = time.time()
    # Load all adjusted data, remove NaNs from adjusted data and sub-sample
    _, adjusted, _ = get_raw_data()
    bool_arr = np.array([not_nan_row(row) for row in adjusted])
    adjusted = adjusted[bool_arr]
    adjusted = sub_sample_data(adjusted, N=img_downsample_threshold)
    
    # split for train and validation data with shuffling
    train, validation = train_test_splity(adjusted, train_split = 0.8)

    # get train input for model:
    train_data = get_stacked_data(train)
    train_angles = get_all_angles(train)
    train_throttles = get_all_throttles(train)
    
    # define additional train weights: data with bigger angle/throttle are weighted more:
    train_angle_weights = (10.0 * np.abs(train_angles).T)[0]
    train_throttle_weights= (10.0 * np.abs(train_throttles).T)[0]
    weights = {'angle_out': train_angle_weights, 'throttle_out': train_throttle_weights}

    # get validation input for model:
    validation_data = get_stacked_data(validation)
    validation_angles = get_all_angles(validation)
    validation_throttles = get_all_throttles(validation)
    
    verbose_level = 0
    if verbose == True:
        print('train_data.shape: \t', train_data.shape)
        print('train_angles.shape: \t', train_angles.shape)
        print('train_throttles.shape: \t', train_throttles.shape)
        print('validation_data.shape: \t', validation_data.shape)
        print('validation_angles.shape:', validation_angles.shape)
        print('validation_throttles.shape:', validation_throttles.shape)
        verbose_level = 2  ## note: makes the jupyter output less readable

    print('Data preprocessing in seconds: ', time.time() - now ) 
        
    model = get_model(get_image(0, train).shape)
    
    ## reduce learning rate when reaching a plateu (30 epochs with no change in the first digit of validation score)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, epsilon=0.1, cooldown = 5, verbose=verbose_level)
    stopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=100, verbose=verbose_level, mode='auto')
    
    print ('Training model..')
    now = time.time()
    
    model.compile(optimizer='adam',
                         loss={'angle_out': 'mse', 'throttle_out': 'mse'},
                         loss_weights={'angle_out': .98, 'throttle_out': .02}) # always full-throttle seems to be fine for now
    
    history = model.fit(train_data, [train_angles, train_throttles], batch_size=64, epochs=EPOCHS, sample_weight=weights,
                        validation_data=(validation_data, [validation_angles, validation_throttles]), verbose=verbose_level, 
                        callbacks=[stopping])
    
    print ('\nModel trained after {:.1f} seconds, final train loss: {:.3f} and validation loss: {:.3f}'.format(
            time.time() - now, history.history['loss'][-1], history.history['val_loss'][-1]))
           
    
    # save the model as needed for the simulator:
    json = model.to_json()
    model.save_weights('model.h5')
    with open('model.json', 'w') as f:
        f.write(json)
    print ('Model saved')
    
    # return history for visualization in jupyter:
    return history


## only run the following code when executed as 'python model.py'
## ( NOT when imported in jupyter ) ;-)
if __name__ == "__main__":
    print('NOTE: Automatic training of the model is disabled when calling \'python model.py\' because the code runs in the jupyter notebook.\n      If you want to execute model.py as stand-alone then please un-comment the last line in model.py')
    history = train_model()
