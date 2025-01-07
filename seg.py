import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.io import loadmat
from sklearn.preprocessing import scale, minmax_scale
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import math


def padding(data):
    sqrt_num = math.sqrt(len(data))
    nearest_len = round(sqrt_num) ** 2
    if nearest_len > len(data):
        pad = np.zeros((abs(len(data) - nearest_len), data.shape[-1]))
        data = np.concatenate((data, pad), axis=0)
    else:
        data = data[:nearest_len]
    return data.reshape((round(sqrt_num), round(sqrt_num), -1))


# img = img_as_float(astronaut()[::2, ::2])

img_p = r'../data/PaviaU.mat'
gt_p = r'../data/PaviaU_gt.mat'

img = loadmat(img_p)['paviaU']
gt = loadmat(gt_p)['paviaU_gt']
shape = img.shape
img = img.reshape(-1, shape[-1])
img = scale(img)
img_aggregation = img.copy()
img_aggregation = img_aggregation.reshape(shape)
pca = PCA(n_components=3)
img = pca.fit_transform(img)
img = minmax_scale(img)
img = img.reshape(shape[0], shape[1], -1)
segments_slic = slic(img, n_segments=5000, compactness=10, sigma=1, start_label=1)
print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
# avg = {}

databank = []
for i in np.unique(segments_slic):
    idxs = np.argwhere(segments_slic == i)
    points = []
    for idx in idxs:
        points.append(img_aggregation[idx[0], idx[1]])
    points = np.asarray(points)
    # print(len(points))
    avg = points.mean(0)
    points = padding(points)
    databank.append(points)
    for idx in idxs:
        img_aggregation[idx[0], idx[1]] = avg

rgbimg = np.zeros((shape[0], shape[1], 3))
rgbimg[:, :, 0] = img_aggregation[:, :, 29]
rgbimg[:, :, 1] = img_aggregation[:, :, 19]
rgbimg[:, :, 2] = img_aggregation[:, :, 9]
rgbimg = rgbimg.reshape(-1, 3)
rgbimg = minmax_scale(rgbimg)
rgbimg = rgbimg.reshape(shape[0], shape[1], -1)
fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax[0, 0].imshow(gt)
ax[0, 0].set_title("Ground truth")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(img)
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(rgbimg)
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

a = 0
