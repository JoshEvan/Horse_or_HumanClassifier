import os
import tensorflow as tf

train_horse_dir = os.path.join('/horse-or-human/horses')
train_human_dir = os.path.join('/horse-or-human/human')
validation_horse_dir = os.path.join('/validation-horse-or-human/horses')
validation_human_dir = os.path.join('/validation-horse-or-human/horses')

print("DESIGNING MODEL")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150,3)),
    # Pooling was done to reduce or compress the size of the image
    # we are dong maximum pooling by window of 2 by 2
    # from 4 pixel value we pick the maximum one and replace the 4 values with the maximum
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    # Hidden Layer with 512 neuron
    tf.keras.layers.Dense(512, activation = 'relu'),
    # Output layer
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])

# printing the history / journey of the data
model.summary()

print("BUILD MODEL")
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
            metrics=['acc'])


print("PREPROCESSING DATA")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
validation_datagen = ImageDataGenerator(rescale=1/255) #normalizing pixel value

train_gen = train_datagen.flow_from_directory(
    os.getcwd()+'/horse-or-human/',
    target_size=(150,150), #size of image input
    batch_size=128,
    class_mode = 'binary'
)

# validation (test) data
validation_gen = validation_datagen.flow_from_directory(
    os.getcwd()+'/validation-horse-or-human',
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'binary'
)


print("TRAINING")
history = model.fit_generator(
    train_gen,
    steps_per_epoch=8,
    epochs=5,
    verbose=1, # mode of viewing training process
    validation_data= validation_gen,
    validation_steps = 8
)

print("RUNNING MODEL")
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

li = os.listdir(os.getcwd()+"/test")
print(li)
print(len(li))
plt.figure()


for idx,f in enumerate(os.listdir(os.getcwd()+'/test')):
    tempf = f
    f = os.getcwd()+"/test/"+f
    # print(f)
    img = image.load_img(f, target_size=(150,150))
    plt.subplot(len(li),len(li),idx+1)
    plt.imshow(img)
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis = 0)
    
    images = np.vstack([img_arr])
    classes = model.predict(images,batch_size=10)
    # print(classes[0])
    plt.yticks([])
    plt.xticks([])
    if( classes[0] > 0.5):
        print(f + ' is a human')
        plt.title(tempf + ' is a human')
    else:
        print(f + ' is a horse')
        plt.title(tempf + ' is a horse')
plt.show()