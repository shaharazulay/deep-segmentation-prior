import numpy as np
from sklearn import cluster


def represent_pixel_using_window(img, window_size):
    pad_width = int((window_size - 1) / 2)
    _, n, m = img.shape
    
    img_padded = np.pad(img, [(0, 0), (pad_width, pad_width), (pad_width, pad_width)], 'constant')
    
    img_window = []
    for i_shift in range(window_size):
        for j_shift in range(window_size):
            img_shift = img_padded[:, i_shift:n + i_shift, j_shift:m + j_shift]
            img_window.append(img_shift)
    img_rep = np.concatenate(img_window, axis=0)
    return img_rep
    
    
def pixel_cluster_to_image_array(pixel_cluster, n, m):
    
    image_arr = pixel_cluster.transpose().reshape(1, n, m)
    return image_arr
    
    
def get_cluster_segmentation(img, window_size, n_clusters=2):
    img_rep = represent_pixel_using_window(img, window_size=window_size)
    d, n, m = img_rep.shape
    
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(img_rep.reshape(d, n * m).transpose())
    
    pixel_cluster = kmeans.predict(img_rep.reshape(d, n * m).transpose())
    
    cluster_image_arr = pixel_cluster_to_image_array(pixel_cluster, n, m)
    return cluster_image_arr
    