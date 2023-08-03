import tqdm
import tensorflow as tf
from config import *
from utils import data_utils, train_utils, io_utils, eval_utils, post_processing, anchor_utils, bbox_utils
from utils.preset import preset


def main():
    model, start_epoch, max_mAP50, max_mAP, max_loss = train_utils.load_model(MODEL_TYPE, ANCHORS, NUM_CLASSES, STRIDES, IOU_THRESHOLD,
                                                                              EPS, INF, KERNEL_INITIALIZER, LOAD_CHECKPOINTS, CHECKPOINTS)
    dataloader = data_utils.DataLoader(DTYPE, LABELS, BATCH_SIZE, ANCHORS, NUM_CLASSES, 
                                       model.input_size, model.strides, POSITIVE_IOU_THRESHOLD, MAX_BBOXES, 
                                       CREATE_ANCHORS)
    train_dataset = dataloader('train')
    valid_dataset = dataloader('val', use_label=True)
    train_dataset_length = dataloader.length('train') // BATCH_SIZE
    valid_dataset_length = dataloader.length('val') // BATCH_SIZE

    train_max_loss = valid_max_loss = max_loss
    
    global_step = (start_epoch-1) * train_dataset_length + 1
    warmup_step = 0
    warmup_max_step = train_dataset_length * WARMUP_EPOCHS
    max_step = EPOCHS * train_dataset_length

    lr_scheduler = train_utils.LR_scheduler(train_dataset_length, max_step, warmup_max_step, LR_SCHEDULER, LR)
    optimizer = tf.keras.optimizers.Adam(decay=0.005)

    train_writer = tf.summary.create_file_writer(LOGDIR)
    val_writer = tf.summary.create_file_writer(LOGDIR)
    
    anchors_xywh = list(map(lambda x: tf.reshape(x, [-1,4]), model.anchors_xywh))

    eval = eval_utils.Eval(LABELS, EPS)
    
    @tf.function
    def train_step(batch_images, batch_grids):
        with tf.GradientTape() as train_tape:
            preds = model(batch_images, True)
            train_loss = model.loss(batch_grids, preds, BATCH_SIZE)
            gradients = train_tape.gradient(train_loss[3], model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return train_loss
    
    @tf.function
    def test_step(batch_images, batch_grids):
        preds = model(batch_images)
        valid_loss = model.loss(batch_grids, preds, BATCH_SIZE)
        
        batch_processed_preds = post_processing.prediction_to_bbox(preds, anchors_xywh, BATCH_SIZE, model.strides, NUM_CLASSES, model.input_size)

        return valid_loss, batch_processed_preds
    
    def update_stats_step(batch_processed_preds, batch_labels):
        for processed_preds, labels in zip(batch_processed_preds, batch_labels):
            NMS_preds = post_processing.NMS(processed_preds, SCORE_THRESHOLD, IOU_THRESHOLD, NMS_TYPE, SIGMA).numpy()
            labels = bbox_utils.extract_real_labels(labels).numpy()
            eval.update_stats(NMS_preds, labels)

        return eval.calculate_mAP()

    for epoch in range(start_epoch, EPOCHS + 1):
        #train
        train_iter, train_loc_loss, train_conf_loss, train_prob_loss, train_total_loss = 0, 0., 0., 0., 0.
            
        train_tqdm = tqdm.tqdm(train_dataset, total=train_dataset_length, desc=f'train epoch {epoch}/{EPOCHS}', ascii=' =', colour='red')
        for batch_data in train_tqdm:
            optimizer.lr.assign(lr_scheduler(global_step, warmup_step))

            batch_images = batch_data[0]
            batch_grids = batch_data[1:]

            train_loss = train_step(batch_images, batch_grids)

            train_loc_loss += train_loss[0]
            train_conf_loss += train_loss[1]
            train_prob_loss += train_loss[2]
            train_total_loss += train_loss[3]

            global_step += 1
            train_iter += 1
            warmup_step += 1
            
            train_loss_ = [train_loc_loss/train_iter, train_conf_loss/train_iter,
                           train_prob_loss/train_iter, train_total_loss/train_iter]
            
            io_utils.write_summary(train_writer, global_step, optimizer.lr.numpy(), train_loss_)
            tqdm_text = f'lr={optimizer.lr.numpy():.7f}, ' +\
                        f'total_loss={train_loss_[3].numpy():.3f}, ' +\
                        f'loc_loss={train_loss_[0].numpy():.3f}, ' +\
                        f'conf_loss={train_loss_[1].numpy():.3f}, ' +\
                        f'prob_loss={train_loss_[2].numpy():.3f}'
            train_tqdm.set_postfix_str(tqdm_text)
            
        if train_loss_[3] < train_max_loss:
            train_max_loss = train_loss_[3]
            train_utils.save_model(model, epoch, 0., 0., train_loss_, TRAIN_CHECKPOINTS_DIR)
                
        # valid
        if epoch % EVAL_PER_EPOCHS == 0:
            eval.init_stat()
            valid_iter, valid_loc_loss, valid_conf_loss, valid_prob_loss, valid_total_loss = 0, 0, 0, 0, 0
            valid_tqdm = tqdm.tqdm(valid_dataset, total=valid_dataset_length, desc=f'valid epoch {epoch}/{EPOCHS}', ascii=' =', colour='blue')
            for batch_data in valid_tqdm:
                batch_images = batch_data[0]
                batch_grids = batch_data[1:-1]
                batch_labels = batch_data[-1]
                
                valid_loss, batch_processed_preds = test_step(batch_images, batch_grids)

                mAP50, mAP = update_stats_step(batch_processed_preds, batch_labels)

                mAP50, mAP = eval.calculate_mAP()
                
                valid_loc_loss += valid_loss[0]
                valid_conf_loss += valid_loss[1]
                valid_prob_loss += valid_loss[2]
                valid_total_loss += valid_loss[3]

                valid_iter += 1
                
                valid_loss_ = [valid_loc_loss / valid_iter, valid_conf_loss / valid_iter, 
                               valid_prob_loss / valid_iter, valid_total_loss / valid_iter]
                
                tqdm_text = f'mAP50={mAP50:.4f}, ' +\
                            f'mAP={mAP:.4f}, ' +\
                            f'total_loss={valid_loss_[3].numpy():.3f}, ' +\
                            f'loc_loss={valid_loss_[0].numpy():.3f}, ' +\
                            f'conf_loss={valid_loss_[1].numpy():.3f}, ' +\
                            f'prob_loss={valid_loss_[2].numpy():.3f}'
                valid_tqdm.set_postfix_str(tqdm_text)
            
            io_utils.write_summary(val_writer, epoch, [mAP50, mAP], valid_loss_, False)
            
            if valid_loss_[3] < valid_max_loss:
                valid_max_loss = valid_loss_[3]
                train_utils.save_model(model, epoch, mAP50, mAP, valid_loss_, LOSS_CHECKPOINTS_DIR)
            if mAP50 > max_mAP50:
                max_mAP50 = mAP50
            if mAP > max_mAP:
                max_mAP = mAP
                train_utils.save_model(model, epoch, mAP50, mAP, valid_loss_, MAP_CHECKPOINTS_DIR)

       
if __name__ == '__main__':
    preset(1)
    main()
