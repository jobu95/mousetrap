import os

os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin')
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/extras/CUPTI/lib64')
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include')
os.add_dll_directory('C:/Users/jevbb/Documents/mousetrap/cuda/bin')

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time

import frames_pb2
import pb_analysis
import pb_io

from IPython import display

BATCH_SIZE = 256
EPOCHS = 100
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 64
MODEL_FILENAME = "mouse_generator.keras"

def save_training_images(data):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    fig = plt.figure(figsize=(8, 8))

    for i in range(NUM_EXAMPLES_TO_GENERATE):
        plt.subplot(8, 8, i+1)
        plt.imshow(data[i, :, :, 0] * 127.5 + 127.5, cmap='gray', vmin=0, vmax=255)
        #plt.imshow(data[i, :, :, 0] * 0 + 127.5, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

    plt.savefig('training_data.png')

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*2*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #
    model.add(layers.Reshape((4, 2, 256)))
    assert model.output_shape == (None, 4, 2, 256)  # Note: None is the batch size
    # Kernel is 4x2. Can't be more than 2 tall since our data is 2 rows.
    # Could be more than 4 wide, but this seemed reasonable.
    kernel_sz = (4, 2)
    model.add(layers.Conv2DTranspose(128, kernel_sz, strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 2, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #
    model.add(layers.Conv2DTranspose(64, kernel_sz, strides=(2, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 2, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #
    model.add(layers.Conv2DTranspose(1, kernel_sz, strides=(2, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 16, 2, 1)
    #
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4, 2), strides=(2, 2), padding='same',
        input_shape=[16, 2, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #
    model.add(layers.Conv2D(128, (8, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    #
    return model

# Helper function to compute cross entropy loss.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(generator, discriminator, images, generator_optimizer,
        discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    #
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        #
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        #
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    #
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    #
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, generator, discriminator, generator_optimizer,
        discriminator_optimizer):
    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, image_batch,
                    generator_optimizer, discriminator_optimizer)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                epoch + 1,
                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def train_arcs(arcs, filename = MODEL_FILENAME):
    arcs = pb_analysis.get_interesting_arcs(arcs)
    arcs = pb_io.center_arcs(arcs)
    arcs = pb_io.unsign_arcs(arcs)
    darcs = pb_io.arcs_abs_to_delta(arcs)
    train_images = pb_io.darcs_to_array(darcs)
    train_images = train_images.reshape(train_images.shape[0], 16, 2, 1).astype('float32')
    print("Dataset size: {}".format(train_images.shape[0]))
    train_dataset =
    tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(BATCH_SIZE)

    save_training_images(train_images)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer,
            discriminator_optimizer)

    generator.save(filename)

def gen_arcs(num_arcs, filename = MODEL_FILENAME):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    generator = tf.keras.models.load_model(filename)

    raw_arcs = generator(noise, training=False)
    for i in range(int(num_arcs / BATCH_SIZE)):
        raw_arcs = np.concatenate((raw_arcs, generator(noise, training=False)))
    print("Number of arcs: {}".format(raw_arcs.shape[0]))

    # TODO load this from a proto
    min_dx = -39
    max_dx = 49
    min_dy = -85
    max_dy = 142
    framerate = 60
    return pb_io.array_to_darcs(raw_arcs, min_dx, max_dx, min_dy, max_dy,
            framerate)

