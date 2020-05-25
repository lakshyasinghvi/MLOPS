from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
model = MobileNet(weights='imagenet'  , include_top = False ,input_shape=(224,224,3))
model.layers[0].input
model.layers[0].__class__.__name__
model.layers[3].trainable = False
for  layer in model.layers:
    layer.trainable = False
top_model=model.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Dense(2048, activation='relu')(top_model)
top_model = Dense(1024, activation='relu')(top_model)
top_model = Dense(512, activation='relu')(top_model)
top_model = Dense(256, activation='relu')(top_model)
top_model = Dense(5, activation='softmax')(top_model)
newmodel= Model(inputs=model.input,outputs=top_model)

train_data_dir = 'CelebData/train/'
validation_data_dir = 'CelebData/val/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True)
 
validation_datagen = ImageDataGenerator(rescale=1./255)

img_rows, img_cols = 224, 224

batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
        
newmodel.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
newmodel.fit(
     train_generator,
    steps_per_epoch = 50,
    epochs = 7,
    validation_data = validation_generator,
    validation_steps = 1)
    
newmodel.save('face_recognition_mobilenet.h5')
classifier = load_model('face_recognition_mobilenet.h5')

CelebData_dict =     {"[0]": "ben_afflek"    , 
                      "[1]": "elton_john"    ,
                      "[2]": "jerry_seinfeld",
                      "[3]": "madonna"       ,
                      "[4]": "mindy_kaling"  }

CelebData_dict_n =    {"ben_afflek": "ben_afflek"    , 
                       "elton_john": "elton_john"    ,
                       "jerry_seinfeld": "jerry_seinfeld",
                       "madonna": "madonna"       ,
                       "mindy_kaling": "mindy_kaling"  }

def draw_test(name, pred, im):
    face = CelebData_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, face, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + CelebData_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("CelebData/val/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()
