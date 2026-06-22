import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import mlflow
import sys

mlflow.set_experiment("coffeebeans-classification")

X_train = np.load('coffeebeans_preprocessing/X_train.npy')
y_train = np.load('coffeebeans_preprocessing/y_train.npy')
X_val = np.load('coffeebeans_preprocessing/X_val.npy')
y_val = np.load('coffeebeans_preprocessing/y_val.npy')
X_test = np.load('coffeebeans_preprocessing/X_test.npy')
y_test = np.load('coffeebeans_preprocessing/y_test.npy')

input_example = X_train[0:5]

# 1. Pastikan autolog diaktifkan DI AWAL sebelum memulai/mengelola run
mlflow.autolog()

# 2. Cukup gunakan ini, MLflow akan otomatis mendeteksi run yang sudah ada 
# (dari CLI) atau memulai yang baru jika tidak ada.
if not mlflow.active_run():
    mlflow.start_run()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95 and logs.get('val_accuracy')>0.95):
            print("\nAkurasi telah mencapai >95%!")
            self.model.stop_training = True
callbacks = myCallback()

n_filters = int(sys.argv[1]) if len(sys.argv) > 1 else 32
filter_size = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (3, 3)
activation_function = str(sys.argv[3]) if len(sys.argv) > 3 else 'relu'
input_shape = tuple(map(int, sys.argv[4].split(','))) if len(sys.argv) > 4 else (150, 150, 1)
dense_1 = int(sys.argv[5]) if len(sys.argv) > 5 else 128
dense_2 = int(sys.argv[6]) if len(sys.argv) > 6 else 64

model_1 = Sequential([
    tf.keras.layers.Conv2D(n_filters, filter_size, activation=activation_function, input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(n_filters, filter_size, activation=activation_function, padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(dense_1, activation=activation_function),
    tf.keras.layers.Dense(dense_2, activation=activation_function),
    tf.keras.layers.Dense(4, activation='softmax')
])

model_1.compile(loss='categorical_crossentropy',
            optimizer=tf.optimizers.Adam(),
            metrics=['accuracy'])

print(model_1.summary())

history_1 = model_1.fit(X_train, y_train,
                        epochs=25,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[callbacks],
                        verbose=1)

print("--- Evaluasi Model pada Data Testing ---")
test_loss, test_accuracy = model_1.evaluate(X_test, y_test)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
