import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Reshape,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    BatchNormalization
)
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# -----------------------------
# Generator
# -----------------------------
def make_generator_model():
    model = keras.Sequential()

    model.add(
        Dense(7 * 7 * 256, activation='relu', input_shape=(NOISE_DIM,))
    )
    model.add(Reshape((7, 7, 256)))

    model.add(
        Conv2DTranspose(
            128, (5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2DTranspose(
            64, (5, 5),
            strides=(2, 2),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2DTranspose(
            1, (5, 5),
            strides=(2, 2),
            padding='same',
            activation='tanh'
        )
    )

    return model


# -----------------------------
# Discriminator
# -----------------------------
def make_discriminator_model():
    model = keras.Sequential()

    model.add(
        Conv2D(
            64, (5, 5),
            strides=(2, 2),
            padding='same',
            input_shape=(28, 28, 1)
        )
    )
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            128, (5, 5),
            strides=(2, 2),
            padding='same'
        )
    )
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1))  # logits

    return model


# -----------------------------
# Loss functions
# -----------------------------
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        tf.ones_like(real_output), real_output
    )
    fake_loss = cross_entropy(
        tf.zeros_like(fake_output), fake_output
    )
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(
        tf.ones_like(fake_output), fake_output
    )


# -----------------------------
# Training step
# -----------------------------
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


# -----------------------------
# Training loop
# -----------------------------
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)


# -----------------------------
# Generate & save images
# -----------------------------
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = tf.reshape(predictions, (-1, 28, 28))

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f"outputs/image_at_epoch_{epoch:04d}.png")
    plt.show()


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":

    # Load and preprocess MNIST
    (train_images, _), (_, _) = mnist.load_data()

    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1
    ).astype("float32")

    train_images = (train_images - 127.5) / 127.5

    # Limit dataset for fast execution
    train_images = train_images[:100]

    # Hyperparameters
    BUFFER_SIZE = 100
    BATCH_SIZE = 4
    NOISE_DIM = 100
    EPOCHS = 2

    # Create models
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Loss & optimizers
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Dataset
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
    )

    # Output directory
    os.makedirs("outputs", exist_ok=True)

    # Train GAN
    train(train_dataset, EPOCHS)

    # Generate sample images
    test_input = tf.random.normal([16, NOISE_DIM])
    generate_and_save_images(generator, EPOCHS, test_input)
