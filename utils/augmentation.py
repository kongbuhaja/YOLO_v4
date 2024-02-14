import tensorflow as tf
import numpy as np

@tf.function
def resize_padding(image, labels, out_size, random=False, mode='CONSTANT', constant_values=0, seed=42):
    size = tf.cast(tf.shape(image)[-2:-4:-1], tf.float32)
    ratio = out_size/tf.reduce_max(size)
    new_size = tf.cast(ratio * size, tf.int32)
    pad_size = out_size - tf.cast(new_size, tf.float32)

    pad_ratio = tf.random.uniform((), 0, 1, seed=seed) if random else 0.5
    pad_LT = tf.cast(pad_size*pad_ratio, tf.int32)
    pad_RB = tf.cast(pad_size, tf.int32) - pad_LT
    pad_left, pad_top = pad_LT[0], pad_LT[1]
    pad_right, pad_bottom = pad_RB[0], pad_RB[1]
    padding = tf.reshape(tf.stack([pad_top, pad_bottom, pad_left, pad_right, 0, 0]), [3, 2])

    resized_image = tf.image.resize(image, new_size[::-1])
    padded_image = tf.pad(resized_image, padding, mode=mode, constant_values=constant_values)
    resized_labels = tf.round(tf.concat([labels[..., 0:1] * ratio[0] + float(pad_left),
                                         labels[..., 1:2] * ratio[1] + float(pad_top),
                                         labels[..., 2:3] * ratio[0] + float(pad_left),
                                         labels[..., 3:4] * ratio[1] + float(pad_top),
                                         labels[..., 4:]], -1))
    return padded_image, resized_labels

def eval_resize_padding(image, out_size, mode='CONSTANT', constant_values=0):
    size = tf.cast(tf.shape(image)[-2:-4:-1], tf.float32)
    ratio = out_size/tf.reduce_max(size)
    new_size = tf.cast(ratio * size, tf.int32)
    pad_size = out_size - tf.cast(new_size, tf.float32)

    pad_LT = tf.cast(pad_size*0.5, tf.int32)
    pad_RB = tf.cast(pad_size, tf.int32) - pad_LT
    pad_left, pad_top = pad_LT[0], pad_LT[1]
    pad_right, pad_bottom = pad_RB[0], pad_RB[1]
    padding = tf.reshape(tf.stack([pad_top, pad_bottom, pad_left, pad_right, 0,0]), [3,2])

    resized_image = tf.image.resize(image, new_size[::-1])
    padded_image = tf.pad(resized_image, padding, mode=mode, constant_values=constant_values)
    
    return padded_image, tf.cast(pad_LT, tf.float32)

@tf.function
def random_augmentation(image, labels, image_size, seed=42):
    bboxes = labels[..., :4]
    
    geometric_methods = [[random_scale, 0.5], [random_rotate90, 0.25], [random_flip_horizontally, 0.5], [random_crop, 0.5]]
    kernel_methods = [[random_gaussian_blur, 0.5]]
    color_methods = [[random_brigthness, 0.5], [random_hue, 0.5], [random_saturation, 0.5], [random_contrast, 0.5]]

    for augmentation_method, prob_threshold in geometric_methods + kernel_methods + color_methods:
        image, bboxes = randomly_apply(augmentation_method, image, bboxes, image_size, prob_threshold, seed=seed)
    labels = tf.concat([bboxes, labels[..., 4:]], -1)

    return minmax(image, labels)

@tf.function
def randomly_apply(method, image, bboxes, image_size, prob_threshold=0.5, seed=42):
    if tf.random.uniform((), seed=seed) > prob_threshold:
        return method(image, bboxes, image_size, seed=seed)
    return image, bboxes

@tf.function
def random_scale(image, bboxes, image_size, lower=0.7, upper=1.0, seed=42):
    size = tf.cast(tf.shape(image)[-2:-4:-1], tf.float32)
    scaled_image_size = tf.random.uniform((2, ), lower, upper, seed=seed) * image_size
    ratio = tf.reduce_min(scaled_image_size / size)
    scaled_size = tf.cast(tf.round(size * ratio), tf.int32)
    scaled_image = tf.image.resize(image, scaled_size[::-1])
    scaled_bboxes = tf.round(bboxes * ratio)
    
    return scaled_image, scaled_bboxes

@tf.function
def random_crop(image, bboxes, image_size, seed=42):
    bbox_LT = tf.reduce_min(bboxes[..., :2], axis=0)
    bbox_RB = tf.reduce_max(bboxes[..., 2:], axis=0)
    
    if tf.reduce_any(bbox_LT >= bbox_RB):
        return image, bboxes
    elif tf.reduce_sum(bboxes) == 0:
        return image, bboxes
    else:
        size = tf.cast(tf.shape(image)[-2:-4:-1], tf.float32)

        crop_LT = tf.cast(tf.random.uniform((2,), seed=seed) * bbox_LT, tf.int32)
        crop_RB = tf.cast(tf.random.uniform((2,), seed=seed) * (size - bbox_RB) + bbox_RB, tf.int32)
        crop_image = image[crop_LT[1]:crop_RB[1], crop_LT[0]:crop_RB[0]]
        bboxes -= tf.cast(tf.tile(crop_LT[None], [1,2]), tf.float32)

        return crop_image, bboxes

@tf.function
def random_flip_horizontally(image, bboxes, image_size, seed=42):
    if tf.reduce_sum(bboxes) != 0:
        size = tf.cast(tf.shape(image)[-2:-4:-1], tf.float32)
        bboxes = tf.stack([size[0] - bboxes[..., 2] - 1,
                        bboxes[..., 1],
                        size[0] - bboxes[..., 0] - 1,
                        bboxes[..., 3]], -1)
    
    return tf.image.flip_left_right(image), bboxes

@tf.function
def random_rotate90(image, bboxes, image_size, seed=42):
    def rotate90_bboxes(bboxes, width, height):
        if tf.reduce_sum(bboxes) != 0:
            bboxes = tf.stack([bboxes[..., 1],
                                width - bboxes[..., 2] -1,
                                bboxes[..., 3],
                                width - bboxes[..., 0] -1], -1)
        return bboxes, height, width
    
    k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32, seed=seed)
    size = tf.cast(tf.shape(image)[-2:-4:-1], tf.float32)
    width, height = size[0], size[1]
    for time in tf.range(k):
        size = tf.shape(image)[-2:-4:-1]
        bboxes, width, height = rotate90_bboxes(bboxes, width, height)
    return tf.image.rot90(image, k), bboxes

@tf.function
def random_gaussian_blur(image, bboxes, image_size, ksize=3, sigma=1, seed=42):
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
    return blur_image, bboxes

@tf.function
def random_saturation(image, bboxes, image_size, lower=0.5, upper=1.5, seed=42):
    return tf.image.random_saturation(image, lower, upper, seed=seed), bboxes

@tf.function
def random_hue(image, bboxes, image_size, max_delta=0.08, seed=42):
    return tf.image.random_hue(image, max_delta, seed=seed), bboxes

@tf.function
def random_contrast(image, bboxes, image_size, lower=0.5, upper=1.5, seed=42):
    return tf.image.random_contrast(image, lower, upper, seed=seed), bboxes

@tf.function
def random_brigthness(image, bboxes, image_size, max_delta=0.12, seed=42):
    return tf.image.random_brightness(image, max_delta, seed=seed), bboxes

@tf.function
def minmax(image, labels):
    return tf.maximum(tf.minimum(image, 1.), 0), labels
