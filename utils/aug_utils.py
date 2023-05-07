import tensorflow as tf
import numpy as np
import cv2

def augmentation(image, labels, width, height):
    bboxes = labels[..., :4]
    color_methods = [random_brigthness, random_contrast, random_hue, random_saturation]
    geometric_methods = [random_flip_horizontally, random_crop]
    for augmentation_method in geometric_methods + color_methods:
        image, bboxes, width, height = randomly_apply(augmentation_method, image, bboxes, width, height)
    labels = tf.concat([bboxes, labels[..., 4:5]], -1)
    image = tf.maximum(tf.minimum(image, 255), 0)
    return image, labels, image.shape[1], image.shape[0]

@tf.function
def tf_augmentation(image, labels, width, height):
    bboxes = labels[..., :4]
    color_methods = [random_brigthness, random_hue, random_saturation]
    geometric_methods = [random_flip_horizontally, random_flip_vertically, random_crop]
    for augmentation_method in geometric_methods + color_methods:
        image, bboxes, width, height = randomly_apply(augmentation_method, image, bboxes, width, height)
    labels = tf.concat([bboxes, labels[..., 4:5]], -1)
    return image, labels, width, height

def resize_padding(image, labels, image_size):
    height, width = image.shape[:2]
    scale = min(image_size/width, image_size/height)
    new_width, new_height = int(scale * width), int(scale * height)
    pad_width, pad_height = (image_size - new_width)//2, (image_size - new_height)//2
    labels[:, [0,2]] = (labels[:, [0,2]] * scale + pad_width).astype(np.int32)
    labels[:, [1,3]] = (labels[:, [1,3]] * scale + pad_height).astype(np.int32)

    image_resized = cv2.resize(image, (new_width, new_height))
    image_pad = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image_pad[pad_height:pad_height+new_height, pad_width:pad_width+new_width] = image_resized

    return image_pad, labels

@tf.function
def tf_resize_padding_tf_function(image, labels, width, height, image_size):
    scale = tf.minimum(image_size/width, image_size/height)
    new_width, new_height = tf.floor(scale * width), tf.floor(scale * height)
    pad_width, pad_height = (image_size - new_width)//2, (image_size - new_height)//2
    labels_ = tf.floor(tf.concat([labels[..., 0:1] * scale + pad_width,
                                  labels[..., 1:2] * scale + pad_height,
                                  labels[..., 2:3] * scale + pad_width,
                                  labels[..., 3:4] * scale + pad_height,
                                  labels[..., 4:5]],-1))
    image_pad = tf.image.resize_with_pad(image, image_size, image_size)
    
    return image_pad, labels_

@tf.function
def tf_resize_padding(image, labels, width, height, image_size):
    scale = tf.minimum(image_size/width, image_size/height)
    new_width, new_height = tf.floor(scale * width), tf.floor(scale * height)
    pad_left, pad_top = (image_size - new_width)//2, (image_size - new_height)//2
    pad_right, pad_bottom = image_size - new_width - pad_left, image_size - new_height - pad_top
    labels_ = tf.floor(tf.concat([labels[..., 0:1] * scale + pad_left,
                                  labels[..., 1:2] * scale + pad_top,
                                  labels[..., 2:3] * scale + pad_left,
                                  labels[..., 3:4] * scale + pad_top,
                                  labels[..., 4:5]],-1))
    image_resized = tf.image.resize(image, [new_height,new_width])
    padding = tf.cast(tf.reshape(tf.stack([pad_top, pad_bottom, pad_left, pad_right, 0,0]),(3,2)), tf.int32)
    image_pad = tf.pad(image_resized, padding)
    return image_pad, labels_

@tf.function
def randomly_apply(method, image, bboxes, width, height):
    if tf.random.uniform(())>0.5:
        return method(image, bboxes, width, height)
    return image, bboxes, width, height

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
    bboxes = tf.stack([width - bboxes[..., 2] - 1,
                       bboxes[..., 1],
                       width - bboxes[..., 0] - 1,
                       bboxes[..., 3]], -1)
    
    return tf.image.flip_left_right(image), bboxes, width, height

@tf.function
def random_flip_vertically(image, bboxes, width, height):
    bboxes = tf.stack([bboxes[..., 0],
                       height - bboxes[..., 3] - 1,
                       bboxes[..., 2],
                       height - bboxes[..., 1] - 1,], -1)
    return tf.image.flip_up_down(image), bboxes, width, height

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
