from torch import nn


class CNN(nn.Module):
    def __init__(self, dim, latent_dim):
        super(CNN, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        self.embedding = nn.Conv1d(self.dim, self.latent_dim, kernel_size=1)
        self.smoothing = nn.Conv1d(
            self.latent_dim, self.latent_dim, kernel_size=13, padding=6
        )
        self.bn = nn.BatchNorm1d(self.latent_dim)
        self.relu = nn.ReLU()
        self.unembedding = nn.Conv1d(self.latent_dim, self.dim, kernel_size=1)

    def forward(self, X):
        X = self.embedding(X)
        X = self.smoothing(X)
        X = self.bn(X)
        X = self.relu(X)
        X = self.unembedding(X)
        return X
