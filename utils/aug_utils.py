import tensorflow as tf
import numpy as np
import cv2

@tf.function
def tf_resize_padding(image, labels, width, height, image_size):
    scale = tf.minimum(image_size/width, image_size/height)
    new_width, new_height = tf.floor(scale * width), tf.floor(scale * height)
    pad_left, pad_top = (image_size - new_width)//2, (image_size - new_height)//2
    pad_right, pad_bottom = image_size - new_width - pad_left, image_size - new_height - pad_top
    resized_image = tf.image.resize(image, [new_height,new_width])
    padding = tf.cast(tf.reshape(tf.stack([pad_top, pad_bottom, pad_left, pad_right, 0,0]), [3,2]), tf.int32)
    padded_image = tf.pad(resized_image, padding)
    resized_labels = tf.round(tf.concat([labels[..., 0:1] * scale + pad_left,
                                         labels[..., 1:2] * scale + pad_top,
                                         labels[..., 2:3] * scale + pad_left,
                                         labels[..., 3:4] * scale + pad_top,
                                         labels[..., 4:]],-1))
    
    return padded_image, resized_labels

@tf.function
def tf_augmentation(image, labels, width, height):
    bboxes = labels[..., :4]
    color_methods = [[random_brigthness, 0.5], [random_hue, 0.5], [random_saturation, 0.5], [random_contrast, 0.5]]
    kernel_methods = [[random_gaussian_blur, 0.5]]
    geometric_methods = [[random_rotate90, 0.25], [random_flip_horizontally, 0.5], [random_scale, 0.5], [random_crop, 0.5]]

    for augmentation_method, prob_threshold in geometric_methods + kernel_methods + color_methods:
        image, bboxes, width, height = randomly_apply(augmentation_method, image, bboxes, width, height, prob_threshold)
    labels = tf.concat([bboxes, labels[..., 4:]], -1)

    return image, labels, width, height

@tf.function
def randomly_apply(method, image, bboxes, width, height, prob_threshold=0.5):
    if tf.random.uniform(()) > prob_threshold:
        return method(image, bboxes, width, height)
    return image, bboxes, width, height

@tf.function
def random_scale(image, bboxes, width, height, lower=0.7, upper=1.3):
    ratio = tf.random.uniform((2,), lower, upper)
    new_width = tf.round(width * ratio[0])
    new_height = tf.round(height * ratio[1])
    scaled_image = tf.image.resize(image, [new_height, new_width])
    scaled_bboxes = tf.round(tf.stack([bboxes[..., 0] * ratio[0],
                                       bboxes[..., 1] * ratio[1],
                                       bboxes[..., 2] * ratio[0],
                                       bboxes[..., 3] * ratio[1]], -1))
    
    return scaled_image, scaled_bboxes, new_width, new_height

@tf.function
def random_crop(image, bboxes, width, height):
    bbox_left_top = tf.reduce_min(bboxes[..., :2], axis=0)
    bbox_right_bottom = tf.reduce_max(bboxes[..., 2:], axis=0)
    
    if tf.reduce_any(bbox_left_top >= bbox_right_bottom):
        return image, bboxes, width, height
    elif tf.reduce_sum(bboxes) == 0:
        return image, bboxes, width, height
    else:
        crop_left = tf.cast(tf.random.uniform(()) * bbox_left_top[0], tf.int32)
        crop_top = tf.cast(tf.random.uniform(()) * bbox_left_top[1], tf.int32)
        crop_right = tf.cast(tf.random.uniform(()) * (width - bbox_right_bottom[0]) + bbox_right_bottom[0], tf.int32)
        crop_bottom = tf.cast(tf.random.uniform(()) * (height - bbox_right_bottom[1]) + bbox_right_bottom[1], tf.int32)
        
        crop_image = image[crop_top:crop_bottom, crop_left:crop_right]

        bboxes -= tf.cast(tf.tile(tf.stack([crop_left, crop_top], -1)[None], [1,2]), tf.float32)
        
        width = tf.cast(crop_right - crop_left, tf.float32)
        height = tf.cast(crop_bottom - crop_top, tf.float32)
        
        return crop_image, bboxes, width, height

@tf.function
def random_flip_horizontally(image, bboxes, width, height):
    bboxes = tf.stack([width - bboxes[..., 0] - 1,
                       bboxes[..., 1],
                       width - bboxes[..., 2] - 1,
                       bboxes[..., 3]], -1)
    
    return tf.image.flip_left_right(image), bboxes, width, height

@tf.function
def random_rotate90(image, bboxes, width, height):
    def rotate90_bboxes(bboxes_, width_, height_):
        new_bboxes_ = tf.stack([bboxes_[..., 1],
                                width_ - bboxes_[..., 0] -1,
                                bboxes_[..., 3],
                                width_ - bboxes_[..., 2]] -1, -1)
        return new_bboxes_, height_, width_
    
    k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)
    for time in tf.range(k):
        bboxes, width, height = rotate90_bboxes(bboxes, width, height)
    return tf.image.rot90(image, k), bboxes, width, height

@tf.function
def random_gaussian_blur(image, bboxes, width, height, ksize=3, sigma=1):
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
    return blur_image, bboxes, width, height

@tf.function
def random_saturation(image, bboxes, width, height, lower=0.5, upper=1.5):
    return tf.image.random_saturation(image, lower, upper), bboxes, width, height

@tf.function
def random_hue(image, bboxes, width, height, max_delta=0.08):
    return tf.image.random_hue(image, max_delta), bboxes, width, height

@tf.function
def random_contrast(image, bboxes, width, height, lower=0.5, upper=1.5):
    return tf.image.random_contrast(image, lower, upper), bboxes, width, height

@tf.function
def random_brigthness(image, bboxes, width, height, max_delta=0.12):
    return tf.image.random_brightness(image, max_delta), bboxes, width, height
