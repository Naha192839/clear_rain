import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50V2,ResNet50,VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,GlobalAveragePooling2D,AveragePooling2D
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import os , datetime,glob,shutil,numpy as np 

# classes = ['sunny','cloudy'] #分類するクラス
classes = ['晴れ曇り','雨'] #分類するクラス
nb_classes = len(classes)

train_data_dir = './dataset_ver2_kfold/train'
validation_data_dir = './dataset_ver2_kfold/val'
test_data_dir = './dataset_ver2_kfold/test_check'

# train_data_dir = './TWI/train'
# validation_data_dir = './TWI/val'
# test_data_dir = './TWI/test'
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
width_shift_range=0.5,
horizontal_flip=True,
zoom_range=0.5
)

# validation
val_datagen = ImageDataGenerator(rescale=1. / 255)

# test 用
test_datagen = ImageDataGenerator(rescale=1. / 255)

num = 0
val_num = 0 
val_count = 0
ep = 50

all_loss=[]
all_val_loss=[]
all_test_loss=[]

all_acc=[]
all_val_acc=[]
all_test_acc=[]
#交差検証
for i in range(8):
  print(val_count)
  for index, classlabel in enumerate(classes):
    val_dir = "./dataset_ver2_kfold/val/" + classlabel
    train_dir = "./dataset_ver2_kfold/train/" + classlabel
    for p in os.listdir(val_dir):
      shutil.move(os.path.join(val_dir, p), train_dir)#検証データを訓練データに移動
    files = glob.glob(train_dir + "/*.jpg")#全データを取得
    val_list = files[val_count:val_count+100]#検証データの確保
    for i in val_list:
      shutil.move(i,val_dir)#検証ディレクトに移動
    
  train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size= train_batch_size,# 1回のバッチ生成で作る画像数
    shuffle=True,
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

  input_tensor = Input(shape=(img_height,img_width, 3))
  # resnet50 = ResNet50V2(include_top=False, weights='imagenet',input_tensor=input_tensor,pooling='avg')
  resnet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor,pooling='avg')

  # 重みパラメータの凍結
  resnet50.trainable = False

  x=resnet50.output
  predictions = Dense(nb_classes, activation='softmax')(x)

  model = Model(resnet50.input, predictions)

  # block5の重みパラメーターを解凍
  for layer in model.layers[:154]:
    layer.trainable = False
  for layer in model.layers[154:]:
    layer.trainable = True
    
  # for layer in model.layers[:15]:
  #   layer.trainable = False
  # for layer in model.layers[15:]:
  #   layer.trainable = True
  model.summary()



  model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
                metrics=['accuracy'])
  plot_model(model, show_shapes=True,show_layer_names=False,to_file='model.png')
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.00001)
  history = model.fit(
    train_generator, 
    steps_per_epoch = train_generator.n // train_batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.n // val_batch_size,
    epochs=ep,
    callbacks=[reduce_lr]
  )

  # Evaluate the model on the test data using `evaluate`
  print("Evaluate on test data")
  results = model.evaluate(
    test_generator,
    steps=test_generator.n // val_batch_size)
  print("test loss, test acc:", results)
  
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



  # Plot training & validation accuracy values
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.ylim(bottom=0.4)
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.savefig(os.path.join("./fig/acc_fig/",str(datetime.datetime.today())+"acc.jpg"))
  plt.clf()

  # Plot training & validation loss values
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.savefig(os.path.join("./fig/loss_fig/",str(datetime.datetime.today())+"loss.jpg"))
  plt.clf()

  model.save('./cle_model_ver2/ResNet50v1_'+str(num)+'_aug.h5')    
  num += 1
  val_count = val_count + 100

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
print(all_test_loss)
print(all_test_acc)
print("ave_all_test_loss"+str(ave_all_test_loss))
print("ave_all_test_acc"+str(ave_all_test_acc))
