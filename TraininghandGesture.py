### 01.Importing libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout ## overfiting prevent 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
##########################################################################

### 02.Design CNN model
model=Sequential() #oreder of layers about the model
model.add(Conv2D(32,(3,3),input_shape=(256,256,1),activation='relu'))#input layer
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(units=150,activation='relu')) #fully connected layer
model.add(Dropout(0.25))# overfitting prevent
model.add(Dense(units=6,activation='softmax')) # Output layer-6 classes
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
##########################################################################

### 03.Pre-processing data
train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=12,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.15,
                                 horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory('HandGestureDataset/train',
                                                 target_size=(256,256),
                                                 color_mode='grayscale',
                                                 batch_size=8,
                                                 classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                                 class_mode='categorical')

val_generator=val_datagen.flow_from_directory('HandGestureDataset/test',
                                              target_size=(256,256),
                                              color_mode='grayscale',
                                              batch_size=8,
                                              classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
                                              class_mode='categorical') 

callback_list=[
    EarlyStopping(monitor='val_loss',patience=10),
    ModelCheckpoint(filepath="model.keras",monitor='val_loss',save_best_only=True,verbose=1)]

model.fit(train_generator,
                    steps_per_epoch=37,
                    epochs=5,
                    validation_data=val_generator,
                    validation_steps=7,
                    callbacks=callback_list 
         )
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")
print("Saved model to disk")
