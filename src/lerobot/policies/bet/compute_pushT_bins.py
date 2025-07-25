import numpy as np
from sklearn.cluster import KMeans
import torch
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/pusht_image")

# collect actions into an (N,2) array
actions = np.stack([data["action"].numpy() for data in ds], axis=0)
print(f"Found {len(actions)} actions in the dataset.")


# Perform k-means clustering to find centroids
nbins = 32
centroids = np.zeros((2, nbins))

km = KMeans(n_clusters=nbins,
             random_state=0,
             n_init=10,
             max_iter=300).fit(actions)

centroids_2d = km.cluster_centers_
centroids = centroids_2d.T  # Transpose to match (2, nbins) shape

print("Found centroids:")
print(centroids)
print("Saving centroids to pushT_bin_centroids.pt")


torch.save(torch.from_numpy(centroids).float(), "pushT_bin_centroids.pt")

# save_path = os.path.join(os.getcwd(), "src/lerobot/policies/bet", "pushT_bin_centroids.pt")

# print("Save path:", save_path)

# torch.save(torch.from_numpy(centroids).float(), save_path)


# import matplotlib.pyplot as plt

# plt.scatter(actions[:,0], actions[:,1], s=1, alpha=0.3)
# plt.scatter(centroids[0, :], centroids[1, :], c='red', marker='x')
# plt.title("Action space & KMeans centroids")
# plt.xlabel("Action dim 1")
# plt.ylabel("Action dim 2")
# plt.show()

