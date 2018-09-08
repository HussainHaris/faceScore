import pandas as pd
import numpy as np 
import tensorflow

def replaceLabel(value):
	if value == -1:
		return 0
	else:
		return 1

label_cols = ['image_file', 'popularity'] + 'partial_faces is_female baby child teenager youth middle_age senior white black asian oval_face round_face heart_face smiling mouth_open frowning wearing_glasses wearing_sunglasses wearing_lipstick tongue_out duck_face black_hair blond_hair brown_hair red_hair curly_hair straight_hair braid_hair showing_cellphone using_earphone using_mirror braces wearing_hat harsh_lighting dim_lighting'.split(' ')
raw_labels = pd.read_csv('selfie_dataset.txt', delim_whitespace=True, names=label_cols)

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

num_images = 2000 # len(all_images)
image_size = 256

expression_cols = ['smiling', 'mouth_open', 'frowning', 'tongue_out', 'duck_face']

all_images_x = np.zeros((num_images, image_size, image_size, 3))
all_images_y = np.zeros((num_images, len(expression_cols)))

for i in range(num_images):
    row = raw_labels.iloc[i]
    
    image_file = 'images/{}.jpg'.format(row['image_file'])
    
    img = image.load_img(image_file, grayscale=False, target_size=(image_size, image_size), interpolation='nearest')
    img_arr = image.img_to_array(img, data_format='channels_last')
    
    all_images_x[i] = img_arr
    all_images_y[i] = [replaceLabel(raw_labels.iloc[i][x]) for x in expression_cols]



    if 100 * (i+1) // num_images > 100 * i // num_images:
    	print('{}% complete'.format(100 * (i+1) / num_images))

datagen = ImageDataGenerator(
	featurewise_center=True,
	featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

datagen.fit(all_images_x)

data_train_flow = datagen.flow(all_images_x, all_images_y, batch_size=64, shuffle=True, subset='training')
data_test_flow = datagen.flow(all_images_x, all_images_y, batch_size=64, shuffle=True, subset='validation')


# Sequential NN
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=(image_size,image_size,3)),
    MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
    
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
    
    Conv2D(256, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
    
    Conv2D(512, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
    
    Conv2D(1024, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
    
    Flatten(),
    
    Dense(1024, activation='relu'),
    Dropout(0.2),
    
    Dense(len(expression_cols), activation='softmax'),
])

model.summary()
import keras.optimizers
model.compile(
    loss='categorical_crossentropy', 
    optimizer=keras.optimizers.Adam(.1), 
    metrics=['accuracy']
)
history = model.fit_generator(data_train_flow, steps_per_epoch=8, epochs=10)
plt.plot(history.history['acc'])


