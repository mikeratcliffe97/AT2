from __future__ import print_function, division

import keras as keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation
from keras.layers import BatchNormalization, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from PIL import Image
import tensorflow as tf
TRAIN_DATA_PATH = os.path.join('dataset/training_data_bin.npy')
TEST_DATA_PATH = os.path.join('dataset/test_data_bin.npy')

session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

class CGAN():
    def __init__(self):
        # Input shape
       
        
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.resolution = 2
        self.num_classes = 10
        self.latent_dim = 100
        
        self.optimizer = Adam(0.0002, 0.5)
#if data does not exist, create
        self.create_bin()
       
    
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=self.optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise and the target label as input
        shapesize = self.latent_dim
        noise = Input(shape=(shapesize,))
        
       # For the combined model we will only train the generator
        self.discriminator.trainable = False
        gen_img = self.generator(noise)
        
        newInput = Input(tensor = gen_img)
       
        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator(gen_img)
        newModel = Model([newInput], valid)
       
      
        self.combined = newModel.compile(optimizer=self.optimizer, loss=["binary_crossentropy"] )
       # self.combined.compile(optimizer=self.optimizer, loss=["binary_crossentropy"] )
    
    def build_generator(self):

      
        model = Sequential()
    
        model.add(Dense(4*4*256,activation="relu",input_dim=self.latent_dim))
        model.add(Reshape((4,4,256)))
    
        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
    
        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
       
        # Output resolution, additional upsampling
        for i in range(self.resolution):
          model.add(UpSampling2D())
          model.add(Conv2D(128,kernel_size=3,padding="same"))
          model.add(BatchNormalization(momentum=0.8))
          model.add(Activation("relu"))
          #model.compile(optimizer)
    
        # Final CNN layer
        model.add(Conv2D(self.channels,kernel_size=3,padding="same"))
       # model.add(Activation("relu"))
    
    
    
        input_shape = Input(shape=(self.latent_dim,))
        generated_image = model(input_shape)
    
        return Model(input_shape,generated_image)

    def build_discriminator(self) -> Model:
    
       
        model = Sequential()

      
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
      
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
    
        model.add(Dropout(0.25))
     
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
    
        model.add(Dropout(0.25))
       
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
    
        model.add(Dropout(0.25))
       
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
    
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, input_shape=self.img_shape))
        model.add(Activation('sigmoid'))
      
        img = Input(shape=self.img_shape)
       
        validity = model(img)
        
        return Model(img, validity)


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        training_data = np.load(TRAIN_DATA_PATH)
        training_data = np.reshape(training_data,(-1,64,64,3))
        training_data = training_data / 127.5 - 1.
        x_train = (training_data.astype(np.float32) - 127.5) / 127.5
        x_real = np.expand_dims(x_train, axis=2)
        y_train = x_real.reshape(-1, 1)
      
 
        # Adversarial ground truths
        y_real = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))
       
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            
            rndInt = np.random.randint(0,x_train.shape[0],batch_size)
            x_real = x_train[rndInt]
            imgs = x_real
          
            # Sample noise as generator input
            noise = np.random.normal(0,1, (batch_size,100))
            # Generate a half batch of new images based on noise
           
            gen_imgs = self.generator.predict(noise)
           
          # Train the discriminator
            dis_loss_real = self.discriminator.train_on_batch([imgs], y_real)
          
            dis_loss_fake = self.discriminator.train_on_batch([gen_imgs], y_fake)
            dis_loss = 0.5 * np.add(dis_loss_real, dis_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator
            random_input = Input(shape=(100,))
            #self.combined = Model([gen_imgs], dis_loss_real)
            gen_loss = self.combined.train_on_batch([random_input] , y_real)
          
                                               
           
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, dis_loss[0], 100*dis_loss[1], gen_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
        
 
    #def load_data(self):

    def create_bin(self):
             
        print("Looking for npy data")

        if not os.path.isfile(TRAIN_DATA_PATH):
            print ("No train data, creating...")
            training_data = []
        
            IMAGE_PATH = os.path.join('dataset/training_set/cat')
            for filename in tqdm(os.listdir(IMAGE_PATH)):
                PATH = os.path.join(IMAGE_PATH, filename)
                image = Image.open(PATH).resize((self.img_cols, self.img_cols), Image.ANTIALIAS) #rows/cols as per gan 
                training_data.append(np.asarray(image, dtype=np.uint8))
            training_data = np.reshape(training_data,(-1, self.img_cols,self.img_cols, self.channels)) #rows cols and channels
            training_data = training_data / 127.5 - 1
               
            np.save(TRAIN_DATA_PATH, training_data)
            print(training_data)
        else:
            training_data = np.load(TRAIN_DATA_PATH)
            
            
            
#        if not os.path.isfile(TEST_DATA_PATH):
#            print ("no test data, creating...")
#            test_data = []
#            IMAGE_PATH = os.path.join('dataset/test_set/cat')
#            for filename in tqdm(os.listdir(IMAGE_PATH)):
#                PATH = os.path.join(IMAGE_PATH, filename)
#                image = Image.open(PATH).resize((28 ,Image.ANTIALIAS)) #rows/cols as per gan 
#                test_data.append(np.asarray(image))
#            test_data = np.reshape(test_data, (-1, 28, 28, 1)) #rows cols and channels
#            test_data = test_data / 127.5 - 1
#            
#            np.save(TEST_DATA_PATH, test_data)
#                
#        else:
#            test_data = np.load(TEST_DATA_PATH)
        
    
    
if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=128, sample_interval=200)