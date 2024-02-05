import tqdm
import tensorflow as tf
from models.model import *
from utils.lr_shcedulers import LR_scheduler
from utils.data_utils import DataLoader
from utils.eval_utils import Eval
from utils.io_utils import read_cfg, Logger

def main():
    cfg = read_cfg()

    epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train']['warmup_epochs']
    train_checkpoint = cfg['model']['train_checkpoint']
    loss_checkpoint = cfg['model']['loss_checkpoint']
    map_checkpoint = cfg['model']['map_checkpoint']
    
    model, start_epoch, max_mAP50, max_mAP, max_loss = load_model(cfg)
    dataloader = DataLoader(cfg)
    
    train_dataset = dataloader('train', augmentation=True)
    valid_dataset = dataloader('val')

    eval = Eval(cfg)
    logger = Logger(cfg)

    train_dataset_length = dataloader.length['train'] // dataloader.batch_size
    valid_dataset_length = dataloader.length['val'] // dataloader.batch_size

    train_best_loss = valid_best_loss = max_loss
    
    global_step = (start_epoch-1) * train_dataset_length + 1
    warmup_step = 0
    warmup_max_step = train_dataset_length * warmup_epochs
    max_step = epochs * train_dataset_length

    lr_scheduler = LR_scheduler(train_dataset_length, 
                                max_step, 
                                warmup_max_step, 
                                cfg['train']['lr_scheduler'], 
                                cfg['train']['lr'])
    optimizer = tf.keras.optimizers.Adam(decay=0.005)
    
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as train_tape:
            preds = model(batch_images, True)
            train_loss = model.loss(batch_labels, preds)
            gradients = train_tape.gradient(train_loss[0], model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return train_loss
    
    @tf.function(experimental_relax_shapes=True)
    def test_step(batch_images, batch_labels):
        preds = model(batch_images)
        valid_loss = model.loss(batch_labels, preds)
        batch_preds = model.output(preds)
        
        return batch_preds, *valid_loss
    
    def update_eval_step(batch_preds, batch_labels):
        for i, preds in enumerate(batch_preds):
            labels = batch_labels[batch_labels[..., 0]==i][..., 1:].numpy()
            NMS_preds = model.NMS(preds).numpy()
            eval.update_stats(NMS_preds, labels)

    for epoch in range(start_epoch, epochs + 1):
        #train
        train_iter, train_total_loss, train_reg_loss, train_obj_loss, train_cls_loss = 0, 0., 0., 0., 0.
            
        train_tqdm = tqdm.tqdm(train_dataset, total=train_dataset_length, ncols=160, desc=f'Train epoch {epoch}/{epochs}', ascii=' =', colour='red')
        for batch_images, batch_labels in train_tqdm:
            optimizer.lr.assign(lr_scheduler(global_step, warmup_step))

            total_loss, reg_loss, obj_loss, cls_loss = train_step(batch_images, batch_labels)

            train_total_loss += total_loss
            train_reg_loss += reg_loss
            train_obj_loss += obj_loss
            train_cls_loss += cls_loss

            global_step += 1
            train_iter += 1
            warmup_step += 1
            
            train_loss = [train_total_loss/train_iter,
                          train_reg_loss/train_iter, 
                          train_obj_loss/train_iter,
                          train_cls_loss/train_iter, ]
            
            logger.write_train_summary(global_step, optimizer.lr.numpy(), train_loss)
            tqdm_text = f'lr={optimizer.lr.numpy():.6f}, ' +\
                        f'total_loss={train_loss[0].numpy():.3f}, ' +\
                        f'reg_loss={train_loss[1].numpy():.3f}, ' +\
                        f'obj_loss={train_loss[2].numpy():.3f}, ' +\
                        f'cls_loss={train_loss[3].numpy():.3f}'

            train_tqdm.set_postfix_str(tqdm_text)
            
        if train_loss[0] < train_best_loss:
            train_best_loss = train_loss[0]
            save_model(model, epoch, 0., 0., train_loss, train_checkpoint)
                
        # valid
        if eval.check(epoch):
            eval.init_stat()
            valid_iter, valid_total_loss, valid_reg_loss, valid_obj_loss, valid_cls_loss = 0, 0, 0, 0, 0
            valid_tqdm = tqdm.tqdm(valid_dataset, total=valid_dataset_length, ncols=160, desc=f'Valid epoch {epoch}/{epochs}', ascii=' =', colour='blue')
            for batch_images, batch_labels in valid_tqdm:
                batch_preds, total_loss, reg_loss, obj_loss, cls_loss = test_step(batch_images, batch_labels)
                update_eval_step(batch_preds, batch_labels)
                mAP50, mAP = eval.calculate_mAP()
                
                valid_total_loss += total_loss
                valid_reg_loss += reg_loss
                valid_obj_loss += obj_loss
                valid_cls_loss += cls_loss

                valid_iter += 1
                
                valid_loss = [valid_total_loss / valid_iter,
                              valid_reg_loss / valid_iter, 
                              valid_obj_loss / valid_iter, 
                              valid_cls_loss / valid_iter]
                
                tqdm_text = f'mAP50={mAP50:.4f}, ' +\
                            f'mAP={mAP:.4f}, ' +\
                            f'total_loss={valid_loss[0].numpy():.3f}, ' +\
                            f'reg_loss={valid_loss[1].numpy():.3f}, ' +\
                            f'obj_loss={valid_loss[2].numpy():.3f}, ' +\
                            f'cls_loss={valid_loss[3].numpy():.3f}'

                valid_tqdm.set_postfix_str(tqdm_text)
            
            logger.write_eval_summary(epoch, mAP50, mAP, valid_loss)
            
            if valid_loss[0] < valid_best_loss:
                if mAP50 > max_mAP50:
                    max_mAP50 = mAP50
                valid_best_loss = valid_loss[0]
                save_model(model, epoch, mAP50, mAP, valid_loss, loss_checkpoint)
            
            if mAP > max_mAP:
                max_mAP = mAP
                save_model(model, epoch, mAP50, mAP, valid_loss, map_checkpoint)
                print(f'\033[32mbest_model is saved with {mAP:.4f} mAP in {epoch} epoch\033[0m')

       
if __name__ == '__main__':
    main()
