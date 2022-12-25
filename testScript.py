from keras.models import load_model
from tqdm import tqdm
import csv  
import os
import numpy as np
import tensorflow as tf


test_dir =r"C:\Users\eslam\OneDrive\Desktop\Testing\Test"
IMG_SIZE = 250

loadedModel = load_model(r'C:\Users\eslam\OneDrive\Desktop\Testing\Xcepiton.h5')
print("Done")

with open('Xception.csv', 'w', encoding='UTF8',newline='') as CsvWriter:
  header = ["image_name","label"]
  writer = csv.writer(CsvWriter)
  writer.writerow(header)
  for img in tqdm(os.listdir(test_dir)):
    path = os.path.join(test_dir, img)
    img_arr = tf.keras.utils.load_img(path,target_size=(IMG_SIZE,IMG_SIZE))
    img_arr = np.asarray(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)
    prediction = loadedModel.predict(img_arr)
    data = [img,np.argmax(prediction)]
    writer.writerow(data)