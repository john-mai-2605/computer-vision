from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import models
import numpy as np
import time
import os
from PIL import Image
import time
# set up network parameters
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 25

# define a function to build a generator
def build_generator():
    model = Sequential()
    model.add(Dense(64 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 64)))
    model.add(UpSampling2D())

    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())

    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    model.summary()
    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img)

# define a function to build a discriminator
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    """
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))"""

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

# build GAN
optimizer = Adam(0.002, 0.95)

# build discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

# build generator
generator = build_generator()
z = Input(shape=(latent_dim,))  # modified from Input(shape=(100,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


# define a function to train GAN
def train(epochs, batch_size=128, save_image_interval=50, save_model_interval = 5000):
    os.makedirs('images', exist_ok=True)
    sub_images = os.listdir("images")
    new_version_sub_path = "images/version" + str(len(sub_images) + 1)
    os.makedirs(new_version_sub_path)

    os.makedirs('saved_model_weights', exist_ok=True)
    num_existing_version_saved = len(os.listdir("saved_model_weights"))
    new_save_model_path = "saved_model_weights/version" + str(num_existing_version_saved + 1)
    os.makedirs(new_save_model_path)

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    list_G_loss = []
    list_D_loss = []
    list_D_acc = []
    list_D_real_image_loss = []
    list_D_fake_image_loss = []

    for epoch in range(epochs):
        print("epoch: ", epoch)
        # Select a random real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        # Sample noise and generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)

        # Train the discriminator
        D_loss_real = discriminator.train_on_batch(real_imgs, valid)
        D_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        D_real_image_loss, D_real_image_acc = D_loss_real
        D_fake_image_loss, D_fake_image_acc = D_loss_fake


        D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid)
        list_D_acc.append(100*D_loss[1])
        list_D_loss.append(D_loss[0])
        list_D_real_image_loss.append(D_real_image_loss)
        list_D_fake_image_loss.append(D_fake_image_loss)
        list_G_loss.append(g_loss)

        # If at save interval
        if epoch % save_image_interval == 0:
            # Print the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, D_loss[0], 100 * D_loss[1], g_loss))
            # Save generated image samples
            save_imgs(new_version_sub_path, epoch)
        if epoch % save_model_interval == 0:
            print("save model of epoch ", epoch)
            generator.save_weights(new_save_model_path + "/generator_weights_" + str(epoch) + ".h5")
            discriminator.save_weights(new_save_model_path + "/discriminator_weights_" + str(epoch) + ".h5")
            combined.save_weights(new_save_model_path + "/combined_weights_"+ str(epoch) + ".h5")

    path_save_plot_training = "plot_history_training/version" + str(len(sub_images) + 1)
    os.makedirs(path_save_plot_training)
    plot_training_process(list_D_acc, list_D_real_image_loss, list_D_fake_image_loss ,list_G_loss, path_save_plot_training +"/plot_training_")

def plot_training_process(D_acc_list, D_real_image_loss_list, D_fake_image_loss_list, G_loss_list, save_path=None):
    plt.plot(D_acc_list)
    plt.title("D_acc")
    if save_path!= None:
        plt.savefig(save_path + "D_acc.png")
    plt.show()

    plt.plot(list(range(1, len(D_real_image_loss_list) + 1)), D_real_image_loss_list, 'b')
    plt.plot(list(range(1, len(D_fake_image_loss_list) + 1)), D_fake_image_loss_list, 'r')
    plt.title("D_loss_on_real_and_fake_image")
    if save_path!= None:
        plt.savefig(save_path + "D_loss_on_real_and_fake_image.png")
    plt.show()

    plt.plot(G_loss_list)
    plt.title("G_loss")
    if save_path!= None:
        plt.savefig(save_path + "G_loss.png")
    plt.show()

def save_imgs(parent_path,epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(parent_path + "/" + "mnist_" + str(epoch) + ".png")
    plt.close()

# train GAN
start = time.time()

train(epochs=35001, batch_size=64, save_image_interval= 500, save_model_interval=5000)

end = time.time()
elapsed_train_time = 'elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                     int((end - start) % 60))
print(elapsed_train_time)



#Image.open('images/mnist_1000.png')
