#
#  The data needs to be in the Data directory
#      Data --
#             Model1
#                 --   Images 1-n
#             Model2
#                 --   Images 1-n
#      Result --
#               cnn_model.keras
# Have the same amount of images in Model1 and Model2
# Otherwise the network prediction will be not accurate
# Train until 90 or more Percent is reached. Otherwise Prediction is not accurate.
#

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Verzeichnis für das Training (hier geht's um das Klassifikationsmodell)
train_dir = "../Data"

# Initilize the ImageDataGenerator mit Data
datagen = ImageDataGenerator(rescale=1./255)

# Load the Training data
train_generator = datagen.flow_from_directory(
    train_dir,               # Path to Dataset
    target_size=(512, 512),  # Image Size change
    batch_size=32,           # Batch-Size
    class_mode='categorical' # Two Classes: Model1, Model2
)



# CNN-Modell erstellen
model_classifier = cnn()

# Kompiliere das Modell
model_classifier.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Trainiere das Modell mit dem train_generator
model_classifier.fit(train_generator, epochs=50, steps_per_epoch=50)
model_classifier.save("../Result/CNN_01" + str(50) + ".keras")

from tensorflow.keras.preprocessing import image
# Load Sample laden
img_path = '/TestImage.jpg'
img = image.load_img(img_path, target_size=(512, 512))

# Convert Image to Array and normilze 
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Dimension for Batch processing add one

# Predict 
prediction = model_classifier.predict(img_array)

# Output der Prediction
print("Prediction (Model1 oder Model2):", prediction)
