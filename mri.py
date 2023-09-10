#IMPORTING RELEVANT PACKAGES
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing import image


train_data_dir = r'C:\Users\goura\Downloads\MRI\Training'
test_data_dir = r'C:\Users\goura\Downloads\MRI\Testing'




import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(r'C:\Users\goura\Downloads\MRI\Training\meningioma\Tr-me_1297.jpg')


img1= mpimg.imread(r'C:\Users\goura\Downloads\MRI\Training\meningioma\Tr-me_1306.jpg')



print(img1.shape)

input_shape = (224,224,3) 
num_classes = 4  


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(input_shape[0],input_shape[1]),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)


score =  model.evaluate(test_generator,verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# Load and preprocess the image you want to classify
# img_path1 = r"C:\Users\goura\Downloads\MRI\Training\glioma\Tr-gl_1299.jpg" # Replace with the path to your image
# img = image.load_img(img_path1, target_size=(224,224))
# img_array1= image.img_to_array(img)
# img_array1 = np.expand_dims(img_array1, axis=0)
# img_array1 = preprocess_input(img_array1)

# # Make predictions using the model

# # img_array1.shape

# # Make predictions using the model
# predictions = model.predict(img_array1)
# # decoded_predictions = decode_predictions(predictions, top=3)[0]

# class_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']
# top_class_index = np.argmax(predictions[0])
# predicted_label = class_labels[top_class_index]
# predicted_labe
model.save('tumor_classification_model1.h5')