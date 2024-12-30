#%%
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from scipy.optimize import minimize
from typing import Tuple
import itertools as it
from skimage.util import img_as_ubyte
import skimage
# %%
# name = "T_S04856"
name = "T_S04923"

png_file = Path(f"../../data/seathru/D3/D3/linearPNG/{name}.png")
tif_file = Path(f"../../data/seathru/D3/D3/depth/depth{name}.tif")
# %%
clusters = 10
# %%
def uint8_2_double(array: np.ndarray):
    return array.astype(np.float64) / 255.0

def double_2_uint8(array: np.ndarray):
    return (array * 255).astype(np.uint8)
# %%
def imshow(img: np.ndarray, color_channel="rgb"):
    if img.dtype == np.float64 and len(img.shape) == 3:
        img = double_2_uint8(img)

    if color_channel == "bgr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_channel == "hsv":
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    #plt.imshow(img)

    #if len(img.shape) == 2:
    #    plt.colorbar()
# %%
png = uint8_2_double(cv2.imread(png_file))
height, width, _ = png.shape

imshow(png, color_channel="bgr")
# %%
png.dtype
# %%
tif_image = Image.open(tif_file)
tif = np.array(tif_image)
tif = cv2.resize(tif, (width, height),  interpolation = cv2.INTER_NEAREST)

imshow(tif)
# %%
png.shape, tif.shape
# %%
tif.min(), tif.max()
# %%
tif_flat = tif.flatten()

tif_flat.shape
# %%
kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(tif_flat[tif_flat != 0].reshape(-1, 1))
labels = np.zeros(tif_flat.shape, dtype=np.uint8)
labels[tif_flat != 0] = kmeans.labels_
labels[tif_flat == 0] = 255 # Use 255 to represent nan
means = np.array(kmeans.cluster_centers_).flatten()
means_args = np.argsort(means)

labels_new = np.zeros_like(labels)
labels_new[labels == 255] = clusters + 1

for current_label in range(clusters):
    new_label = np.nonzero(means_args == current_label)[0]
    labels_new[labels == current_label] = new_label

labels = labels_new.reshape((height, width))
imshow(labels)
# %%
dark_pixels = []
z_values = []

for i in range(clusters):
    mask = labels == i
    filtered = png * mask[:, :, np.newaxis]

    b = png[:, :, 0]
    g = png[:, :, 1]
    r = png[:, :, 2]

    b_filtered = b[mask].flatten()
    g_filtered = g[mask].flatten()
    r_filtered = r[mask].flatten()

    pixels = np.array([[b,g,r] for b,g,r in zip(b_filtered, g_filtered , r_filtered)])
    idx = np.nonzero(np.all(pixels <= np.percentile(pixels, 1, axis=0), axis=1))
    selected_pixels = pixels[idx]

    filtered_tif = tif[mask].flatten()
    z = filtered_tif[idx]

    dark_pixels.extend(selected_pixels.tolist())
    z_values.extend(z.tolist())

dark_pixels = np.array(dark_pixels)
z_values = np.array(z_values)
# %%
dark_pixels.shape, z_values.shape
# %%
tif.min(), tif.max()
# %%
z_values
# %%
def estimate_backscatter(B_inf: float, beta_B: float, J_prime: float, beta_D_prime: float, z):
    return B_inf * (1 - np.exp(- beta_B * z)) + (J_prime * np.exp(- beta_D_prime * z))
# %%
def optimize_estimate_backscatter(arguments: Tuple[float, float, float, float], B_hat: np.ndarray, z: np.ndarray):
    B_inf, beta_B, J_prime, beta_D_prime = arguments

    return np.sum((B_hat - estimate_backscatter(B_inf, beta_B, J_prime, beta_D_prime, z)) ** 2)
# %%
result = minimize(optimize_estimate_backscatter, [1, 1, 1, 1], args=(dark_pixels[:, 0], z_values), bounds=[(0,1), (0,5), (0,1), (0,5)], method="SLSQP")
b_args = result.x

b_args
# %%
optimize_estimate_backscatter(b_args, dark_pixels[:, 0], z_values)
# %%
result = minimize(optimize_estimate_backscatter, [1, 1, 1, 1], args=(dark_pixels[:, 1], z_values), bounds=[(0,1), (0,5), (0,1), (0,5)], method="SLSQP")
g_args = result.x

g_args
# %%
optimize_estimate_backscatter(g_args, dark_pixels[:, 1], z_values)
# %%
result = minimize(optimize_estimate_backscatter, [1, 1, 1, 1], args=(dark_pixels[:, 2], z_values), bounds=[(0,1), (0,5), (0,1), (0,5)], method="SLSQP")
r_args = result.x

r_args
# %%
optimize_estimate_backscatter(r_args, dark_pixels[:, 2], z_values)
# %%
B_b = estimate_backscatter(b_args[0], b_args[1], b_args[2], b_args[3], tif_flat).reshape((height, width))

imshow(B_b)
# %%
B_g = estimate_backscatter(g_args[0], g_args[1], g_args[2], g_args[3], tif_flat).reshape((height, width))

imshow(B_g)
# %%
B_r = estimate_backscatter(r_args[0], r_args[1], r_args[2], r_args[3], tif_flat).reshape((height, width))

imshow(B_r)
# %%
backscatter = np.zeros((height, width, 3), dtype=np.float64)

backscatter[:, :, 0] = B_b
backscatter[:, :, 1] = B_g
backscatter[:, :, 2] = B_r

imshow(backscatter / backscatter.max())
# %%
direct_signal = png - backscatter

sig_min_b = direct_signal[:, :, 0].min()
sig_min_g = direct_signal[:, :, 1].min()
sig_min_r = direct_signal[:, :, 2].min()

if sig_min_b > 0:
    sig_min_b = 0

if sig_min_g > 0:
    sig_min_g = 0

if sig_min_r > 0:
    sig_min_r = 0

direct_signal[:, :, 0] = direct_signal[:, :, 0] - sig_min_b
direct_signal[:, :, 1] = direct_signal[:, :, 1] - sig_min_g
direct_signal[:, :, 2] = direct_signal[:, :, 2] - sig_min_r

plt.imshow(direct_signal)
# %%
epsilon_percent = 0.1
convergence_threshold = 0.001
f = 2.0
# %%
epsilon = (tif.max() - tif_flat[tif_flat != 0].min()) * epsilon_percent

epsilon
# %%
X = np.array([(int(y), int(x), tif[y, x]) for y, x in it.product(np.arange(height), np.arange(width))])

X
# %%
tif_rebuilt = np.zeros_like(tif)
count, _ = X.shape

for i in range(count):
    tif_rebuilt[int(X[i, 0]), int(X[i, 1])] = X[i, 2]

imshow(tif_rebuilt)
# %%
def metric(a: np.ndarray, b: np.ndarray):
    if np.linalg.norm(a[:2] - b[:2]) > 1:
        return np.inf
    
    return np.linalg.norm(a[2] - b[2])
# %%
print("Coffee Time")
dbscan = DBSCAN(eps=epsilon, min_samples=2, n_jobs=-1, metric=metric).fit(X)

dbscan.labels_
# %%
print("No more coffee")
np.savez_compressed("dbscan", dbscan.labels_)