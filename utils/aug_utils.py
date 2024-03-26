import tensorflow as tf
import numpy as np

@tf.function
def batch_augmentation(image, labels, cfg, seed=42):
    geometric_methods = [[crop, cfg['crop']], [rotate90, cfg['rotate90']], [flip_horizontally, cfg['flip_horizontally']]]
    kernel_methods = [[gaussian_blur, cfg['gaussian_blur']]]
    color_methods = [[brightness, cfg['brightness']], [hue, cfg['hue']], [saturation, cfg['saturation']], [contrast, cfg['contrast']]]

    for augmentation_method, prob_threshold in geometric_methods + kernel_methods + color_methods:
        image, labels = randomly_apply(augmentation_method, image, labels, prob_threshold, seed=seed)
    
    return minmax(image, labels)

@tf.function
def mosaic_augmentation(image, labels, cfg, seed=42):
    geometric_methods = [[rotate90, cfg['rotate90']], [flip_horizontally, cfg['flip_horizontally']]]
    kernel_methods = [[gaussian_blur, cfg['gaussian_blur']]]
    color_methods = [[brightness, cfg['brightness']], [hue, cfg['hue']], [saturation, cfg['saturation']], [contrast, cfg['contrast']]]

    for augmentation_method, prob_threshold in geometric_methods + kernel_methods + color_methods:
        image, labels = randomly_apply(augmentation_method, image, labels, prob_threshold, seed=seed)

    return minmax(image, labels)


# @tf.function
def randomly_apply(method, image, labels, prob_threshold=0.5, seed=42):
    if tf.random.uniform((), seed=seed) < prob_threshold:
        return method(image, labels, seed=seed)
    return image, labels
    
# @tf.function
def crop(image, labels, xyxy=None, seed=42):
    if xyxy is None:
        size = tf.cast(tf.shape(image)[:2][::-1], tf.float32)
        x1, y1 = tf.unstack(tf.cast(tf.random.uniform([2], minval=[0,0], maxval=size//3, seed=seed), tf.int32))
        x2, y2 = tf.unstack(tf.cast(tf.random.uniform([2], minval=size//3*2, maxval=size, seed=seed), tf.int32))
    else:
        x1, y1, x2, y2 = tf.unstack(tf.cast(xyxy, tf.int32))
    
    crop_image = image[y1:y2, x1:x2]
    filter = tf.logical_and(tf.logical_and(tf.reduce_all(labels[..., 2:4] > [x1, y1], -1),
                                           tf.reduce_all(labels[..., :2] < [x2, y2], -1)),
                            tf.logical_and(tf.reduce_all((labels[..., 2:3] - labels[..., 0:1]) > 5, -1),
                                           tf.reduce_all((labels[..., 3:4] - labels[..., 1:2]) > 5, -1)))
    filtered_labels = labels[filter]
    crop_labels = tf.concat([tf.maximum(filtered_labels[..., :2] - [x1, y1], 0),
                             filtered_labels[..., 2:4] - [x1, y1] - tf.maximum(filtered_labels[..., 2:4] - [x2, y2], 0),
                             filtered_labels[..., 4:]], -1)
    return crop_image, crop_labels

# @tf.function
def flip_horizontally(image, labels, seed=42):
    if tf.reduce_sum(labels) != 0:
        size = tf.cast(tf.shape(image)[:2][::-1], tf.float32)
        labels = tf.stack([size[0] - labels[..., 2] - 1,
                        labels[..., 1],
                        size[0] - labels[..., 0] - 1,
                        labels[..., 3],
                        labels[..., 4]], -1)
    
    return tf.image.flip_left_right(image), labels

# @tf.function
def rotate90(image, labels, seed=42):
    times = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32, seed=seed)
    width, height = tf.unstack(tf.cast(tf.shape(image)[:2][::-1], tf.float32))

    if times==1:
        labels = tf.stack([labels[..., 1],
                            width - labels[..., 2] -1,
                            labels[..., 3],
                            width - labels[..., 0] -1,
                            labels[..., 4]], -1)
    elif times==2:
        labels = tf.stack([width - labels[..., 2] -1,
                            height - labels[..., 1] -1,
                            width - labels[..., 0] -1,
                            height - labels[..., 3] -1,
                            labels[..., 4]], -1)
    else:
        labels = tf.stack([height - labels[..., 3] -1,
                            labels[..., 0],
                            height - labels[..., 1] -1,
                            labels[..., 2],
                            labels[..., 4]], -1)

    return tf.image.rot90(image, times), labels

# @tf.function
def gaussian_blur(image, labels, sigma=1, seed=42):
    idx = tf.random.uniform((), minval=0, maxval=6, dtype=tf.int32, seed=seed)
    ksize = tf.constant([3, 5, 7, 9, 11, 13])[idx]

    def gaussian_kernel(size=3, sigma=1):
        x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
        y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs**2 + ys**2) / (2*(sigma**2))) / (2*np.pi*(sigma**2))
        return tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)
    kernel = gaussian_kernel(ksize, sigma)[..., None, None]
    
    r, g, b = tf.split(image, [1,1,1], -1)
    r_blur = tf.nn.conv2d(r[None], kernel, [1,1,1,1], 'SAME')
    g_blur = tf.nn.conv2d(g[None], kernel, [1,1,1,1], 'SAME')
    b_blur = tf.nn.conv2d(b[None], kernel, [1,1,1,1], 'SAME')

    blur_image = tf.concat([r_blur, g_blur, b_blur], -1)[0]
    return blur_image, labels

# @tf.function
def saturation(image, labels, lower=0.5, upper=1.5, seed=42):
    return tf.image.random_saturation(image, lower, upper, seed=seed), labels

# @tf.function
def hue(image, labels, max_delta=0.08, seed=42):
    return tf.image.random_hue(image, max_delta, seed=seed), labels

# @tf.function
def contrast(image, labels, lower=0.5, upper=1.5, seed=42):
    return tf.image.random_contrast(image, lower, upper, seed=seed), labels

# @tf.function
def brightness(image, labels, max_delta=0.12, seed=42):
    return tf.image.random_brightness(image, max_delta, seed=seed), labels

# @tf.function
def minmax(image, labels):
    return tf.maximum(tf.minimum(image, 1.), 0), labels


# @tf.function
def resize_padding(image, labels, out_size, random=False, mode='CONSTANT', constant_values=0, seed=42):
    size = tf.cast(tf.shape(image)[:2][::-1], tf.float32)
    ratio = out_size/tf.reduce_max(size)
    new_size = tf.cast(ratio * size, tf.int32)
    pad_size = out_size - tf.cast(new_size, tf.float32)

    pad_ratio = tf.random.uniform((), 0, 1, seed=seed) if random else 0.5
    pad_LT = tf.cast(pad_size*pad_ratio, tf.int32)
    pad_RB = tf.cast(pad_size, tf.int32) - pad_LT
    pad_left, pad_top = tf.unstack(pad_LT)
    pad_right, pad_bottom = tf.unstack(pad_RB)
    padding = tf.reshape(tf.stack([pad_top, pad_bottom, pad_left, pad_right, 0, 0]), [3, 2])

    resized_image = tf.image.resize(image, new_size[::-1])
    padded_image = tf.pad(resized_image, padding, mode=mode, constant_values=constant_values)
    mult = tf.stack([*tf.unstack(ratio),*tf.unstack(ratio),1.])
    add = tf.cast(tf.stack([pad_left, pad_top, pad_left, pad_top, 0]), tf.float32)
    resized_labels = tf.round(labels * mult + add)

    return padded_image, resized_labels

# @tf.function
def resize(image, labels, out_size):
    size = tf.cast(tf.shape(image)[:2][::-1], tf.float32)
    ratio = out_size/tf.reduce_max(size)
    new_size = tf.cast(ratio * size, tf.int32)
    resized_image = tf.image.resize(image, tf.cast(new_size, tf.int32)[::-1])
    mult = tf.stack([*tf.unstack(ratio),*tf.unstack(ratio),1.])
    resized_labels = tf.round(labels * mult)

    return resized_image, resized_labels

def resize_padding_without_labels(image, out_size, mode='CONSTANT', constant_values=0):
    size = tf.cast(tf.shape(image)[:2][::-1], tf.float32)
    ratio = out_size/tf.reduce_max(size)
    new_size = tf.cast(ratio * size, tf.int32)
    pad_size = out_size - tf.cast(new_size, tf.float32)

    pad_ratio = 0.5
    pad_LT = tf.cast(pad_size*pad_ratio, tf.int32)
    pad_RB = tf.cast(pad_size, tf.int32) - pad_LT
    pad_left, pad_top = tf.unstack(pad_LT)
    pad_right, pad_bottom = tf.unstack(pad_RB)
    padding = tf.reshape(tf.stack([pad_top, pad_bottom, pad_left, pad_right, 0, 0]), [3, 2])

    resized_image = tf.image.resize(image, new_size[::-1])
    padded_image = tf.pad(resized_image, padding, mode=mode, constant_values=constant_values)
    
    return padded_image, tf.cast(pad_LT, tf.float32), ratio