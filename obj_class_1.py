import numpy as np 
import pandas as pd 
import os
from pathlib import Path
from int_rect import get_iou
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.layers import Input,Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

data_jpg=[]
data_xml=[]
train_images=[]
train_labels=[]
path=r'C:\Users\sounak\Downloads\reasearch\vgg_app_1\apple'

for filename in os.listdir(path):
    if filename.split('.')[1] == 'jpg':
        data_jpg.append(Path(path,filename))
    else :
        data_xml.append(Path(path,filename))

data_jpg.sort()
data_xml.sort()

# 정렬잘됐나 확인
print(data_jpg[0],data_xml[0])
from xml.etree.ElementTree import parse
object_xmin=[]
object_ymin=[]
object_xmax=[]
object_ymax=[]
gt_values=[]

for i in range(len(data_xml)):
    root_=parse(data_xml[i])
    root=root_.getroot()
    objects = root.findall("object")
    object_xmin.append([int(x.find("bndbox").findtext("xmin")) for x in objects])
    object_ymin.append([int(x.find("bndbox").findtext("ymin")) for x in objects])
    object_xmax.append([int(x.find("bndbox").findtext("xmax")) for x in objects])
    object_ymax.append([int(x.find("bndbox").findtext("ymax")) for x in objects])


#print(object_xmin[i])
import cv2
import matplotlib.pyplot as plt
for i in range(len(data_jpg)):
    img = cv2.imread(str(data_jpg[i]))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for j in range(len(object_xmin[i])):
        #print(object_ymin[i][j])
        #gt_values[0]=object_xmin[i][j]
        ##gt_values[1]=object_ymin[i][j]
        #gt_values[2]=object_xmax[i][j]
        #gt_values[3]=object_ymax[i][j]
        #gt_values.append({"x1":object_xmin[i][j],object_ymin[i][j],object_xmax[i][j],object_ymax[i][j]}
        gt_values=({"x1":object_xmin[i][j],"x2":object_xmax[i][j],"y1":object_ymin[i][j],"y2":object_ymax[i][j]})
        #print(gt_values)
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    imout = img.copy()
    counter = 0
    falsecounter = 0
    flag = 0
    fflag = 0
    bflag = 0
    for e,result in enumerate(ssresults):
        if e < 2000 and flag == 0:
            for gtval in gt_values:
                x,y,w,h = result
                #print(x,y,x+w,y+h)
                #gt=[]
                gt=({"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                #print(gt)
                iou = get_iou(gt_values,gt)
                if counter < 10:
                    if iou > 0.70:
                        timage = img[y:y+h,x:x+w]
                        resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                        cv2.imshow("Cropped Image",resized)
                        cv2.waitKey(0)
                        #cv2.imwrite('cropped.png',resized)
                        #plt.imshow((resized))
                        #plt.show
                        train_images.append(resized)
                        train_labels.append(1)
                        counter += 1
                    
                    else :
                        fflag =1
                if falsecounter <10:
                    if iou < 0.3:
                        timage = img[y:y+h,x:x+w]
                        resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(0)
                        falsecounter += 1
                                
                    else :
                        bflag = 1
            if fflag == 1 and bflag == 1:
                print("inside")
                flag = 1
                #plt.imshow((train_images))
                #plt.show()
X_new = np.array(train_images)
y_new = np.array(train_labels)

dropout1=Dropout(0.5)
dropout2=Dropout(0.5)
dropout3=Dropout(0.5)
model_vgg16_conv=VGG16(weights='imagenet', include_top=False)
#model_vgg16_conv.summary()
input=Input(shape=(224,224,3),name='image_input')
output_vgg16_conv=model_vgg16_conv(input)
x=Flatten(name='flatten')(output_vgg16_conv)
#x=Flatten(name='flatten')(output_vgg16_conv)
x=Dense(2048,activation='relu',name='fc1')(x)
x=dropout1(x)
#x=(Dropout(0.5))
##my_model.add(Dropout(0.5))
x=Dense(1024,activation='relu',name='fc2')(x)
x=dropout2(x)
#my_model.add_loss(Dropout(0.5))
#x=Dense(1024,activation='relu',name='fc3')(x)
x=Dense(512,activation='relu',name='fc3')(x)
x=dropout3(x)
x=Dense(2,activation='softmax',name='predictions')(x)
my_model=Model(inputs=input,outputs=x)
opt = Adam(lr=0.0001)
my_model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
#model_final.summary()

my_model.summary()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y
    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
lenc = MyLabelBinarizer()
Y =  lenc.fit_transform(y_new)
X_train, X_test , y_train, y_test = train_test_split(X_new,Y,test_size=0.10)
trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
hist = my_model.fit_generator(generator= traindata, steps_per_epoch= 10, epochs= 1000, validation_data= testdata, validation_steps=2, callbacks=[checkpoint,early])


                    
                
            
