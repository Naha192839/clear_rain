import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.applications import VGG16, ResNet50V2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D, Cropping2D, ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
import datetime
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.model_selection import KFold

classes = ['晴れ曇り', '雨']  # 分類するクラス
nb_classes = len(classes)

train_data_dir = './dataset/train'
validation_data_dir = './dataset/val'
test_data_dir = './dataset/test_check'
model_dir = "./model"

nb_train_samples = 1400
nb_validation_samples = 200
nb_test_samples = 400

img_width, img_height = 224, 224

train_batch_size = 64
val_batch_size = 16

npz = np.load("./dataset_kfold/clear_rain.npz", allow_pickle=True)
npz_aug = np.load("./dataset_kfold/clear_rain_aug.npz", allow_pickle=True)
# 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
X_train = npz['arr_0'].astype("float") / 255
X_train_aug = npz_aug['arr_0'].astype("float") / 255

# to_categorical()にてラベルをone hot vector化
y_train = np_utils.to_categorical(npz['arr_1'], nb_classes)

print(len(X_train))
print(len(X_train_aug))
print(len(y_train))
# test 用
test_datagen = ImageDataGenerator(rescale=1. / 255)


test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_height, img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size  # 1回のバッチ生成で作る画像数
  )


def three_generator_multiple(generator, dir1, dir2, dir3, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height, img_width),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          classes=classes,
                                          batch_size=batch_size,

                                          seed=7
                                          )
    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height, img_width),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          classes=classes,
                                          batch_size=batch_size,

                                          seed=7
                                          )
    genX3 = generator.flow_from_directory(dir3,
                                          target_size=(img_height, img_width),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          classes=classes,
                                          batch_size=batch_size,

                                          seed=7
                                          )
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            # Yield both images and their mutual label
            yield [X1i[0], X2i[0], X3i[0]], X1i[1]


three_test_generator = three_generator_multiple(test_datagen,
                                          dir1=test_data_dir,
                                          dir2=test_data_dir,
                                          dir3=test_data_dir,
                                          batch_size=val_batch_size,
                                          img_height=img_height,
                                          img_width=img_width)
# 2入力用


def two_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height, img_width),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          classes=classes,
                                          batch_size=batch_size,

                                          seed=7
                                          )
    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height, img_width),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          classes=classes,
                                          batch_size=batch_size,

                                          seed=7
                                          )
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            # Yield both images and their mutual label
            yield [X1i[0], X2i[0]], X1i[1]


two_test_generator = two_generator_multiple(test_datagen,
                                          dir1=test_data_dir,
                                          dir2=test_data_dir,
                                          batch_size=val_batch_size,
                                          img_height=img_height,
                                          img_width=img_width)

kf = KFold(n_splits=8, shuffle=True)  # データ1600枚を８等分して検証データ200枚を確保する
all_loss = []
all_val_loss = []
all_test_loss = []

all_acc = []
all_val_acc = []
all_test_acc = []

num = 0
for train_index, val_index in kf.split(X_train, y_train):
    train_data = X_train[train_index]
    train_label = y_train[train_index]
    val_data = X_train[val_index]
    val_label = y_train[val_index]
    global_input_tensor = Input(shape=(img_height, img_width, 3))
    top_input_tensor = Input(shape=(img_height, img_width, 3))
    bottom_input_tensor = Input(shape=(img_height, img_width, 3))


    # ----------------------------------------------------
    global_model = ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=global_input_tensor)
    global_model.trainable = False

    # block5の重みパラメーターを解凍
    for layer in global_model.layers[:154]:
        layer.trainable = False
    for layer in global_model.layers[154:]:
        layer.trainable = True

    global_model = Model(global_input_tensor,
                         global_model.output, name="global_model")

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
    input_model1 = global_model
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
    # model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, min_lr=0.00001)

    plot_model(model, show_shapes=True,show_layer_names=False,to_file='model.png')
    
    ep = 50
    
    # 2入力用
    if input_model3 == 0:
            history = model.fit(
            [X_train,X_train],
            y_train, 
            batch_size=64,
            # steps_per_epoch = nb_train_samples // train_batch_size, #こいつのためにtrain_generatorを残している
            validation_data = ([val_data,val_data],val_label),
            # validation_steps = nb_validation_samples // val_batch_size,
            epochs=ep,
            # callbacks=[reduce_lr]
    )

            # Evaluate the model on the test data using `evaluate`
            print("Evaluate on test data")
            results = model.evaluate(
                two_test_generator,
                steps= nb_test_samples // val_batch_size)
            print("test loss, test acc:", results)
            model.save('./cle_model/global_top_aug_'+str(num)+'.h5')
            num += 1

    #  3入力用
    else:
            history = model.fit(
            [X_train,X_train,X_train],
            y_train,
            batch_size=64,
            # steps_per_epoch = nb_train_samples // train_batch_size, #こいつのためにtrain_generatorを残している
            validation_data = ([val_data,val_data,val_data],val_label),
            # validation_steps = nb_validation_samples // val_batch_size,
            epochs=ep
            # callbacks=[reduce_lr]
        )
        # Evaluate the model on the test data using `evaluate`
            print("Evaluate on test data")
            results = model.evaluate(
                three_test_generator,
                steps=nb_test_samples // val_batch_size)
            print("test loss, test acc:", results)
            model.save('./cle_model/3C-CNN_aug_'+str(num)+'.h5')
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