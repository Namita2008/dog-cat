import tensorflow as tf
# Load images
data = tf.keras.utils.image_dataset_from_directory(
    "D:/namita/dog-cat-class (3)/dog-cat-class/train",
    image_size=(64, 64),
    batch_size=2
)
# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation="softmax")
])
# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Train
model.fit(data, epochs=50)
model.summary()
# Prediction
img = tf.keras.utils.load_img("D:/namita/dog-cat-class (3)/dog-cat-class/test/p1.jpg", 
                              target_size=(64,64))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)

p = model.predict(x)
print(data.class_names[p.argmax()])
model.save('CAT.keras')
m=tf.keras.models.load_model("cat.keras")
img = tf.keras.utils.load_img(r"D:\namita\dog-cat-class (3)\dog-cat-class\test\p5.jpeg" ,
                              target_size=(64,64))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)
p=m.predict(x)
print(p)
print(data.class_names[p.argmax()])
