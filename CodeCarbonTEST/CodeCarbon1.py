from codecarbon import EmissionsTracker

# Initialize the tracker with custom parameters
tracker = EmissionsTracker(
    # measure_power_secs=10.0,
    # output_dir="results/",
    project_name="MyMLProject",
    # country_iso_code="USA",
    # cloud_provider="AWS",
    # cloud_region="us-east-1",
    # on_cloud=True,
    # log_level="debug"
)

# # Start the tracker
tracker.start()

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

# mnistの形状[28, 28, 1]を定義
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
# generatorが画像を生成するために入力させてあげるノイズの次元
z_dim = 100

# generator(生成器）の定義するための関数
def build_generator(img_shape, z_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(28*28*1, activation='tanh'))
    model.add(Reshape(img_shape))
    return model

# discriminator（識別器）の定義するための関数
def build_discriminatior(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Ganのモデル定義する(生成器と識別器をつなげる)ための関数
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# 実際にGANのモデルをコンパイル
discriminator = build_discriminatior(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
generator = build_generator(img_shape, z_dim)

# 識別器の学習機能をオフに
discriminator.trainable = False 

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

losses = []
accuracies = []
iteration_checkpoint = []

# 学習のための関数
def train(iterations, batch_size, sample_interval):
    (x_train, _), (_, _) = mnist.load_data()
    
    # データの正規化 [-1, 1] にスケール
    x_train = x_train / 127.5 - 1
    x_train = np.expand_dims(x_train, axis=3)

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        # 本物の画像を使用して識別器の訓練
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        # 偽の画像を生成して識別器の訓練
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, acc = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 生成器の訓練
        z = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(z, real)

        # sample_intervalごとに損失値と精度を表示
        if (iteration + 1) % sample_interval == 0:
            print(f"{iteration + 1} [D loss: {d_loss}, acc.: {100 * acc}] [G loss: {g_loss}]")
            losses.append((d_loss, g_loss))
            accuracies.append(acc)
            iteration_checkpoint.append(iteration + 1)
            sample_images(generator)

# 画像を生成するための関数
def sample_images(generator, image_grid_rows=4, image_grid_colmuns=4):
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_colmuns, z_dim))
    gen_images = generator.predict(z)

    # 生成画像を0~1にスケール
    # gen_images = 0.5 * gen_images + 0.5  

    # fig, axs = plt.subplots(image_grid_rows, image_grid_colmuns, figsize=(4, 4), sharex=True, sharey=True)

    # cnt = 0
    # for i in range(image_grid_rows):
    #     for j in range(image_grid_colmuns):
    #         axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap='gray')
    #         axs[i, j].axis('off')
    #         cnt += 1

    # plt.show()  # 画像を表示

# 学習を開始
train(iterations=10000, batch_size=64, sample_interval=1000)

# Stop the tracker
tracker.stop()
