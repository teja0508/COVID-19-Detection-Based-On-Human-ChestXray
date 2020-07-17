# COVID-19-Detection-Based-On-Human-ChestXray

### by Chandrateja Reddy

This is the project that we finished after 12th week of studying **Machine Learning** from scratch.

<p align="center">
<img width="880" height="450" src="https://s3.amazonaws.com/dsg.files.app.content.prod/gereports/wp-content/uploads/2017/05/01182345/lungs.gif">
</p>

## INTRODUCTION
### 1. Motivation
Chest radiography is the most common imaging examination globally, critical for screening, diagnosis, and management of many life threatening diseases. 

My aim is to provide automated COVID-19 chest radiology analysis **comparable to professional practicing radiologists**. In addition, I hope to assist various medical settings such as **improved workflow prioritization** and **clinical decision support**. I have faith that my application will **advance global population health initiatives** through large-scale screening.

### 2. Plan
Main steps:

|Task|Time|Progress|
|:-----------------------------|:------------:|:------------:|
|Data collection |1 days|x|
|Data preprocessing |1 days|x|
|Building Model|5 days|x|
|Build Flask App|1 day|x|
|**Total**|**8 days**||

## MATERIALS AND METHODS
### 1. Datasets
This dataset contains 1000 images of COVID-19 and Normal of 200 patients.I had Provided the COVID-19 dataset,Which all preprocessing done...!!
dataset link: https://drive.google.com/drive/folders/1sIm7jJ_OcTIeNYfx_Yu54c_5whN0-UVL?usp=sharing

### 2. Methods
* **Python** and some neccessary libraries such **Tensorflow, keras, pandas, numpy, keras, tensorflow, flask**.
* **Google Cloud Platform** to train models.

### 3. Building Models
- The training labels in the dataset for each observation are either **0** (COVID +ve), **1** (Normal). Explore different approaches to using the uncertainty labels during the model training.

   
 Binary classification CNN model to recognise 2 different medical observations corona +ve or -ve.
 

* Building **CNN model**

The architecture is built by Tensorflow and Transfer Learning techniques. More details can be found in `COVID-19_detector.ipynb`.

```python
densenet = tf.keras.applications.densenet.DenseNet121(weights='imagenet',input_shape=(224,224,3),include_top=False)
densenet.trainable=False
                                                  
model = tf.keras.Sequential([
    densenet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(14, activation = 'sigmoid')])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=["accuracy"])
```

### 4. Model performance summary

Our model has the AUC of **~97 %** for the train dataset and **~90 %** for the validation dataset. 

### 5. UI
**Pages**
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m1.PNG)
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m2.PNG)
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m3.PNG)
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m4.PNG)
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m5.PNG)
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m6.PNG)
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m7.PNG)
![alt text](https://github.com/teja0508/COVID-19-Detection-Based-On-Human-ChestXray/blob/master/app/static/img/background/m8.PNG)







## CONCLUSION

Successfully **built a deep neural network model** by implementing **Convolutional Neural Network (CNN)** to automatically interprete ChestX-ray_Based-COVID detection.
In addition, we also **built a Flask application** so user can upload their X-ray images and interpret the results.
