import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50V2,ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,Activation,Cropping2D,GlobalAveragePooling2D
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
import os , datetime ,numpy as np
from sklearn.model_selection import KFold

classes = ["晴れ曇り", "雨"]
num_classes = len(classes)
img_width =224
img_height =224

npz = np.load("./dataset_ver2/clear_rain_aug.npz", allow_pickle=True)
# 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
X_train = npz['arr_0'].astype("float") / 255
# to_categorical()にてラベルをone hot vector化
y_train = np_utils.to_categorical(npz['arr_1'], num_classes)


# test 用
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
  './dataset_ver2/test_check',
  target_size=(img_height,img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=16 # 1回のバッチ生成で作る画像数
  )

kf = KFold(n_splits=8, shuffle=True)#データ1600枚を８等分して検証データ200枚を確保する
all_loss=[]
all_val_loss=[]
all_test_loss=[]

all_acc=[]
all_val_acc=[]
all_test_acc=[]

ep=50
num = 0
for train_index, val_index in kf.split(X_train,y_train):
    train_data=X_train[train_index]
    train_label=y_train[train_index]
    val_data=X_train[val_index]
    val_label=y_train[val_index]

    input_tensor = Input(shape=(img_height,img_width, 3))
    # --------------------------------------------------------------
    resnet50 = ResNet50V2(include_top=False, weights='imagenet',input_tensor=input_tensor,pooling='avg')

    # 重みパラメータの凍結
    resnet50.trainable = False

    x=resnet50.output
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(resnet50.input, predictions)

    # block5の重みパラメーターを解凍
    for layer in model.layers[:154]:
        layer.trainable = False
    for layer in model.layers[154:]:
        layer.trainable = True
    # -----------------------------------------------------------
    # model = Sequential()
    # model.add(Cropping2D(cropping=((0,112), (0,0)),input_shape=(img_height, img_width, 3)))
    # model.add(ZeroPadding2D(padding=(3, 3)))
    # model.add(Conv2D(32, (7, 7),strides=(1, 2)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D(padding=(1,1)))
    # model.add(MaxPooling2D((3, 3),strides=(2, 2)))

    # model.add(ZeroPadding2D(padding=(1,1)))
    # model.add(Conv2D(128, (3, 3),strides=(2, 2),activation='relu'))

    # model.add(ZeroPadding2D(padding=(1,1)))
    # model.add(Conv2D(256,(3, 3),strides=(2, 2), activation='relu'))

    # model.add(ZeroPadding2D(padding=(1,1)))
    # model.add(Conv2D(512, (3, 3),strides=(2, 2), activation='relu'))


    # model.add(Conv2D(1024,(3, 3),padding='same',activation='relu'))

    # model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(2, activation='softmax'))
 #-------------------------------------------------------------------------------------------------- 
    # model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])

    history = model.fit(
    X_train, 
    y_train,
    batch_size=64,
    validation_data = (val_data,val_label),
    epochs=ep,
    # callbacks=[reduce_lr,early_stopping]
    )

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(
    test_generator,
    steps= 400 // 16
    )
    print("test loss, test acc:", results)

    model.save('./cle_model_ver2/ResNet50_aug_'+str(num)+'.h5')
    num += 1
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    

    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']

    all_loss.append(loss)
    all_val_loss.append(val_loss)
    all_test_loss.append(results[0])

    all_acc.append(acc)
    all_val_acc.append(val_acc)
    all_test_acc.append(results[1])

ave_all_loss=[
    np.mean([x[i] for x in all_loss]) for i in range(ep)]
ave_all_val_loss=[
    np.mean([x[i] for x in all_val_loss]) for i in range(ep)]

ave_all_acc=[
    np.mean([x[i] for x in all_acc]) for i in range(ep)]
ave_all_val_acc=[
    np.mean([x[i] for x in all_val_acc]) for i in range(ep)]

ave_all_test_loss=[
    np.mean(all_test_loss)]
ave_all_test_acc=[
    np.mean( all_test_acc)]

print("ave_all_loss"+str(ave_all_loss))
print("ave_all_acc"+str(ave_all_acc))
print("ave_all_val_loss"+str(ave_all_val_loss))
print("ave_all_val_acc"+str(ave_all_val_acc))
print(all_test_acc)
print(all_test_loss)
print("ave_all_test_loss"+str(ave_all_test_loss))
print("ave_all_test_acc"+str(ave_all_test_acc))

