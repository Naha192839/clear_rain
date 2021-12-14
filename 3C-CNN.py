import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers,optimizers,regularizers
from tensorflow.keras.applications import VGG16,ResNet50V2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,GlobalAveragePooling2D,AveragePooling2D,Cropping2D,ZeroPadding2D,Conv2D,MaxPooling2D,BatchNormalization,concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
import numpy as np
from keras.utils import np_utils
import os , datetime

classes = ['晴れ曇り','雨'] #分類するクラス
# classes = ['sunny','cloudy'] #分類するクラス
nb_classes = len(classes)

# train_data_dir = './dataset/train'
# validation_data_dir = './dataset/val'
# test_data_dir = './dataset/test_check'

# train_data_dir = './TWI/train'
# validation_data_dir = './TWI/val'
# test_data_dir = './TWI/test'

train_data_dir = './dataset_ver2/train'
validation_data_dir = './dataset_ver2/val'
test_data_dir = './dataset_ver2/test_check'


model_dir = "./model"

nb_train_samples = 1400
nb_validation_samples = 200
nb_test_samples = 400

img_width, img_height = 224, 224

train_batch_size = 64
val_batch_size = 16


# train用
train_datagen = ImageDataGenerator(rescale=1. / 255,
rotation_range=15,
width_shift_range=100,
horizontal_flip=True,
)

# validation
val_datagen = ImageDataGenerator(rescale=1. / 255)

# test 用
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_height, img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size= train_batch_size,# 1回のバッチ生成で作る画像数
  # shuffle=True,
  )
 
validation_generator = val_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_height,img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size,# 1回のバッチ生成で作る画像数
  )

test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_height,img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size # 1回のバッチ生成で作る画像数
  )

def three_generator_multiple(generator,dir1, dir2,dir3, batch_size, img_height,img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          color_mode='rgb',
                                          class_mode = 'categorical',
                                          classes=classes,
                                          batch_size = batch_size,
                                         
                                          seed = 7
                                          )
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          color_mode='rgb',
                                          class_mode = 'categorical',
                                          classes=classes,
                                          batch_size = batch_size,
                                                                                 
                                          seed = 7
                                          )
    genX3 = generator.flow_from_directory(dir3,
                                          target_size = (img_height,img_width),
                                          color_mode='rgb',
                                          class_mode = 'categorical',
                                          classes=classes,
                                          batch_size = batch_size,
                                          
                                          seed = 7
                                          )                                          
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            yield [X1i[0], X2i[0],X3i[0]], X1i[1]  #Yield both images and their mutual label
three_train_generator=three_generator_multiple(train_datagen,
                                           dir1=train_data_dir,
                                           dir2=train_data_dir,
                                           dir3=train_data_dir,
                                           batch_size=train_batch_size,
                                           img_height=img_height,
                                           img_width=img_width)       
     
three_validation_generator=three_generator_multiple(val_datagen,
                                          dir1=validation_data_dir,
                                          dir2=validation_data_dir,
                                          dir3=validation_data_dir,
                                          batch_size=val_batch_size,
                                          img_height=img_height,
                                          img_width=img_width)          

three_test_generator=three_generator_multiple(test_datagen,
                                          dir1=test_data_dir,
                                          dir2=test_data_dir,
                                          dir3=test_data_dir,
                                          batch_size=val_batch_size,
                                          img_height=img_height,
                                          img_width=img_width)
# 2入力用
def two_generator_multiple(generator,dir1, dir2,batch_size, img_height,img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          color_mode='rgb',
                                          class_mode = 'categorical',
                                          classes=classes,
                                          batch_size = batch_size,
                                                                                  
                                          seed = 7
                                          )
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          color_mode='rgb',
                                          class_mode = 'categorical',
                                          classes=classes,
                                          batch_size = batch_size,
                                    
                                          seed = 7
                                          )                                      
    while True:
            X1i = genX1.next()
            X2i = genX2.next()    
            yield [X1i[0], X2i[0]], X1i[1]  #Yield both images and their mutual label
            
two_train_generator=two_generator_multiple(train_datagen,
                                           dir1=train_data_dir,
                                           dir2=train_data_dir,                                    
                                           batch_size=train_batch_size,
                                           img_height=img_height,
                                           img_width=img_width)       
     
two_validation_generator=two_generator_multiple(val_datagen,
                                          dir1=validation_data_dir,
                                          dir2=validation_data_dir,                                          
                                          batch_size=val_batch_size,
                                          img_height=img_height,
                                          img_width=img_width)          

two_test_generator=two_generator_multiple(test_datagen,
                                          dir1=test_data_dir,
                                          dir2=test_data_dir,                                          
                                          batch_size=val_batch_size,
                                          img_height=img_height,
                                          img_width=img_width)

global_input_tensor = Input(shape=(img_height,img_width,3))
top_input_tensor = Input(shape=(img_height,img_width,3))
bottom_input_tensor = Input(shape=(img_height,img_width,3))

# ----------------------------------------------------
global_model = ResNet50V2(include_top=False, weights='imagenet',input_tensor=global_input_tensor)
global_model.trainable = False

# block5の重みパラメーターを解凍
for layer in global_model.layers[:154]:
   layer.trainable = False
for layer in global_model.layers[154:]:
   layer.trainable = True

global_model = Model(global_input_tensor, global_model.output,name="global_model")

# 上領域---------------------------------------------------
top_model = Cropping2D(cropping=((0, 112), (0, 0)))(top_input_tensor)
top_model = ZeroPadding2D(padding=(3, 3))(top_model)
top_model = Conv2D(32, (7, 7), strides=(
    1, 2), activation='relu')(top_model)
top_model = ZeroPadding2D(padding=(1, 1))(top_model)
top_model = MaxPooling2D((3, 3), strides=(2, 2))(top_model)

top_model = ZeroPadding2D(padding=(1,1))(top_model)
top_model = Conv2D(128, (3, 3), strides=(
    2, 2),activation='relu')(top_model)

top_model = ZeroPadding2D(padding=(1, 1))(top_model)
top_model = Conv2D(256, (3, 3), strides=(
    2, 2), activation='relu')(top_model)

top_model = ZeroPadding2D(padding=(1, 1))(top_model)
top_model = Conv2D(512, (3, 3), strides=(
    2, 2), activation='relu')(top_model)

top_model = Conv2D(1024, (3, 3),padding="same",
                    activation='relu')(top_model)
top_model = Model(top_input_tensor, top_model, name="top_model")

# -----------------------------------------------------

# 下領域 ---------------------------------------------------
bottom_model = Cropping2D(cropping=((112, 0), (0, 0)))(bottom_input_tensor)
bottom_model = ZeroPadding2D(padding=(3, 3))(bottom_model)
bottom_model = Conv2D(32, (7, 7), strides=(
    1, 2), activation='relu')(bottom_model)
bottom_model = ZeroPadding2D(padding=(1, 1))(bottom_model)
bottom_model = MaxPooling2D((3, 3), strides=(2, 2))(bottom_model)

bottom_model = ZeroPadding2D(padding=(1,1))(bottom_model)
bottom_model = Conv2D(128, (3, 3),strides=(2,2),
                      activation='relu')(bottom_model)

bottom_model = ZeroPadding2D(padding=(1, 1))(bottom_model)
bottom_model = Conv2D(256, (3, 3), strides=(
    2, 2), activation='relu')(bottom_model)

bottom_model = ZeroPadding2D(padding=(1, 1))(bottom_model)
bottom_model = Conv2D(512, (3, 3), strides=(
    2, 2), activation='relu')(bottom_model)


bottom_model = Conv2D(1024, (3, 3),padding="same", activation='relu')(bottom_model)
bottom_model = Model(bottom_input_tensor,
                      bottom_model, name="bottom_model")
# -----------------------------------------------------
# ここを変えるときはモデル名も変更していください
input_model1 = bottom_model
input_model2 = top_model
input_model3 = 0
if input_model3 == 0:
  model = concatenate([input_model1.output,input_model2.output])
else:
  model = concatenate([input_model1.output,input_model2.output,input_model3.output])

model = GlobalAveragePooling2D()(model)
prediction = Dense(nb_classes, activation='softmax')(model)

if input_model3 == 0:
  model = Model([input_model1.input,input_model2.input],prediction)
else:  
  model = Model([input_model1.input,input_model2.input,input_model3.input],prediction)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])
model.summary()

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=3, min_lr=0.00001)

plot_model(model, show_shapes=True,show_layer_names=False,to_file='model.png')

# 2入力用
if input_model3 == 0:
  history = model.fit(
    two_train_generator, 
    steps_per_epoch = nb_train_samples // train_batch_size, #こいつのためにtrain_generatorを残している
    validation_data = two_validation_generator,
    validation_steps = nb_validation_samples // val_batch_size,
    epochs=100,
    # callbacks=[reduce_lr]
  )
  

  # Evaluate the model on the test data using `evaluate`
  print("Evaluate on test data")
  results = model.evaluate(
    two_test_generator,
    steps= nb_test_samples // val_batch_size)
  print("test loss, test acc:", results)

# 3入力用
else:
  history = model.fit(
    three_train_generator, 
    steps_per_epoch = nb_train_samples // train_batch_size, #こいつのためにtrain_generatorを残している
    validation_data = three_validation_generator,
    validation_steps = nb_validation_samples // val_batch_size,
    epochs=50,
    # callbacks=[reduce_lr]
  )

  # Evaluate the model on the test data using `evaluate`
  print("Evaluate on test data")
  results = model.evaluate(
    three_test_generator,
    steps=nb_test_samples // val_batch_size)
  print("tst loss, test acc:", results)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(os.path.join("./fig/acc_fig/",str(datetime.datetime.today())+"acc.jpg"))
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(os.path.join("./fig/loss_fig/",str(datetime.datetime.today())+"loss.jpg"))

if input_model3 == 0:
  model.save('./cle_model_ver2/top_bottom_aug_2.h5')
else:  
  model.save('./cle_model_ver2/3C-CNN_aug_5.h5')    