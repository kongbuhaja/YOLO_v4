import tensorflow as tf
from losses.common import *

@tf.function
def v4_prob_loss(pred_prob, label_prob, resp_mask, inf, eps):
    prob_loss = resp_mask * binaryCrossEntropy(label_prob, pred_prob, eps)
    prob_loss = tf.reduce_sum(tf.minimum(prob_loss, inf), [1,2,3,4])

    return prob_loss

@tf.function
def v3_prob_loss(pred_prob, label_prob, resp_mask, inf, eps):    
    prob_loss = resp_mask * binaryCrossEntropy(label_prob, pred_prob, eps)
    prob_loss = tf.reduce_sum(tf.minimum(prob_loss, inf), [1,2,3,4])
    
    return prob_loss

def v2_prob_loss(label_cprob, pred_prob_raw, inf, eps):
    resp_mask = label_cprob[..., :1]
    label_prob = label_cprob[..., 1:]
    
    pred_prob = tf.nn.softmax(pred_prob_raw, -1)
    
    prob_loss = resp_mask * tf.square(label_prob - pred_prob)
    prob_loss = tf.reduce_sum(tf.minimum(prob_loss, inf), [1,2,3,4])
    
    return prob_loss