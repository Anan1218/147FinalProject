import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt


class MusicFeaturesDataset(Dataset):
    def __init__(self, df):
        if isinstance(df, pd.DataFrame):
            features = df.values.astype('float32')
        self.features = torch.tensor(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class Generator(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, 128),
      nn.LeakyReLU(0.01),
      nn.Linear(128, 256),
      nn.LeakyReLU(0.01),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.01),
      nn.Linear(512, output_dim),
      nn.Tanh()
    )
      
  def forward(self, z):
    return self.net(z)

class Discriminator(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.LeakyReLU(0.01),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.01),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
      
  def forward(self, x):
    return self.net(x)

def read_input(file_path):
    data = pd.read_csv(file_path, index_col=0)

    categorical_columns = ['artist_name', 'track_name', 'genre']
    numerical_columns = ['popularity', 'year', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                         'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
                         'tempo', 'duration_ms', 'time_signature']
    

    data = data[categorical_columns + numerical_columns]

    encoders = {}

    for column in categorical_columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = data[column].fillna('NA')
            data[column] = le.fit_transform(data[column])
            encoders[column] = le  # store the encoder for potential inverse transformation later

    data[numerical_columns] = data[numerical_columns].astype('float32')

    return data, encoders

def make_plots(discriminator_loss, generator_loss):
  plt.figure(figsize=(10,5))
  plt.plot(discriminator_loss, label='Discriminator Loss')
  plt.plot(generator_loss, label='Generator Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Losses')
  plt.legend()
  plt.show()

def main():
  dataroot = '../data/spotify_data.csv'
  encoders = {}

  features, encoders = read_input(dataroot)

  dataset = MusicFeaturesDataset(features)
  dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

  # for batch in dataloader:
  #   real_features = batch  # Assuming batch directly gives you the features
  #   print("Batch shape:", real_features.shape)  # This should print (128, 18) or similar

  # hyperparameters
  z_dim = 100  # size of the generator input
  feature_dim = features.shape[1]  # size of the features
  lr = 0.0002

  generator = Generator(z_dim, feature_dim)
  discriminator = Discriminator(feature_dim)
  g_optimizer = Adam(generator.parameters(), lr=lr)
  d_optimizer = Adam(discriminator.parameters(), lr=lr)
  loss_fn = nn.BCELoss()

  #plotting
  generator_loss = []
  discriminator_loss = []

  epochs = 200
  for epoch in range(epochs):
    for real_features in dataloader:

      # print("Real features shape before discriminator:", real_features.shape)
      real_pred = discriminator(real_features)
      # print("Real pred shape:", real_pred.shape)


      # real_features = real_features[0]
      batch_size = real_features.size(0)

      real_labels = torch.ones(batch_size, 1, device=real_features.device)
      fake_labels = torch.zeros(batch_size, 1, device=real_features.device)
      # print("Real labels shape:", real_labels.shape)

      d_optimizer.zero_grad()

      # print("Real features shape:", real_features.shape)
      real_pred = discriminator(real_features)
      # print("Real pred shape after discriminator:", real_pred.shape)
      real_loss = loss_fn(real_pred, real_labels)


      z = torch.randn(batch_size, z_dim, device=real_features.device)
      fake_features = generator(z)
      fake_pred = discriminator(fake_features).view(-1, 1)  # Reshape to ensure shape consistency
      fake_loss = loss_fn(fake_pred, fake_labels)

      discriminator_loss = real_loss + fake_loss
      discriminator_loss.backward()
      d_optimizer.step()

      g_optimizer.zero_grad()

      fake_pred = discriminator(fake_features.detach())
      generator_loss = loss_fn(fake_pred, real_labels)
      generator_loss.backward()
      g_optimizer.step()

    discriminator_loss.append(discriminator_loss.item())
    generator_loss.append(generator_loss.item())

    print(f'Epoch [{epoch+1}/{epochs}] | D Loss: {discriminator_loss.item():.4f} | G Loss: {generator_loss.item():.4f}')


  make_plots(discriminator_loss, generator_loss)

main()