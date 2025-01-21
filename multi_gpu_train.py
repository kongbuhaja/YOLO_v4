import tqdm
import tensorflow as tf
from models.model import *
from utils.lr_utils import LR_scheduler
from utils.data_utils import DataLoader
from utils.eval_utils import Eval
from utils.io_utils import read_cfg, Logger
from utils.opt_utils import Optimizer


def main():
    cfg = read_cfg()

    gpus = cfg['gpus']
    epochs = cfg['train']['epochs']

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model, start_epoch, max_mAP = load_model(cfg)
        dataloader = DataLoader(cfg)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = strategy.experimental_distribute_dataset(dataloader('train', cfg['batch_size'], aug=cfg['aug']).with_options(options))
        valid_dataset = strategy.experimental_distribute_dataset(dataloader('val', cfg['eval']['batch_size']).with_options(options))
        optimizer = Optimizer(cfg['train']['optimizer'])

    eval = Eval(cfg)
    logger = Logger(cfg)

    train_dataset_length = dataloader.length['train'] // cfg['batch_size']
    valid_dataset_length = dataloader.length['val'] // cfg['eval']['batch_size']
    
    global_step = (start_epoch-1) * train_dataset_length + 1

    
    lr_scheduler = LR_scheduler(cfg['train']['lr_scheduler'],
                                epochs,
                                train_dataset_length)
    

    with strategy.scope():
        @tf.function(experimental_relax_shapes=True)
        def train_step(batch_images, batch_grids):
            with tf.GradientTape() as train_tape:
                preds = model(batch_images, True)
                train_loss = model.loss(batch_grids, preds)
                gradients = train_tape.gradient(train_loss[0], model.trainable_variables)
                optimizer(zip(gradients, model.trainable_variables))
            
            return train_loss
        
        @tf.function(experimental_relax_shapes=True)
        def eval_step(batch_images, batch_grids):
            preds = model(batch_images)
            valid_loss = model.loss(batch_grids, preds)
            batch_processed_preds = model.decoder.bbox_decode(preds)
            return valid_loss, batch_processed_preds
        
        @tf.function(experimental_relax_shapes=True)
        def distributed_train_step(dist_batch_images, dist_batch_labels):
            dist_replica_losses = strategy.run(train_step, args=(dist_batch_images, dist_batch_labels))
            train_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, dist_replica_losses, axis=None)
            return train_loss
        
        @tf.function(experimental_relax_shapes=True)
        def distributed_eval_step(dist_batch_images, dist_batch_labels):
            dist_replica_losses, dist_batch_preds = strategy.run(eval_step, args=(dist_batch_images, dist_batch_labels))
            test_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, dist_replica_losses, axis=None)          

            return dist_batch_preds, *test_loss
        
        def update_eval_step(dist_batch_preds, dist_batch_labels):
            def step(batch_preds, batch_labels):
                for i, preds in enumerate(batch_preds):
                    try:
                        labels = batch_labels[batch_labels[..., 0]==i][..., 1:].numpy()
                    except:
                        print(batch_labels)
                    NMS_preds = model.decoder.NMS(preds.numpy())
                    eval.update(labels, NMS_preds)
            if gpus == 1:
                step(dist_batch_preds, dist_batch_labels)
            else:
                for gpu in range(gpus):
                    step(dist_batch_preds.values[gpu], dist_batch_labels.values[gpu])
                        
        for epoch in range(start_epoch, epochs + 1):  
            if cfg['aug']['mosaic'] and epoch==cfg['train']['mosaic_epochs']:
                cfg['aug']['mosaic'] = 0
                train_dataset = dataloader('train', cfg['batch_size'], aug=cfg['aug'])
                train_dataset_length = dataloader.length['train'] // cfg['batch_size']

            #train
            train_iter, train_total_loss, train_reg_loss, train_obj_loss, train_cls_loss = 0, 0., 0., 0., 0.
                
            train_tqdm = tqdm.tqdm(train_dataset, total=train_dataset_length, ncols=160, desc=f'Train epoch {epoch}/{epochs}', ascii=' =', colour='red')
            for dist_batch_images, dist_batch_labels in train_tqdm:
                optimizer.assign_lr(lr_scheduler(global_step))

                total_loss, reg_loss, obj_loss, cls_loss = distributed_train_step(dist_batch_images, dist_batch_labels)
                
                train_total_loss += total_loss
                train_reg_loss += reg_loss
                train_obj_loss += obj_loss
                train_cls_loss += cls_loss

                global_step += 1
                train_iter += 1
                
                train_loss = [train_total_loss/train_iter,
                            train_reg_loss/train_iter, 
                            train_obj_loss/train_iter,
                            train_cls_loss/train_iter, ]

                logger.write_train_summary(global_step, optimizer.lr, train_loss)
                tqdm_text = f'lr={optimizer.lr:.6f}, ' +\
                            f'total_loss={train_loss[0].numpy():.3f}, ' +\
                            f'reg_loss={train_loss[1].numpy():.3f}, ' +\
                            f'obj_loss={train_loss[2].numpy():.3f}, ' +\
                            f'cls_loss={train_loss[3].numpy():.3f}'
                
                train_tqdm.set_postfix_str(tqdm_text)
                        
            # valid
            if eval.check(epoch):
                eval.init_stat()
                valid_iter, valid_total_loss, valid_reg_loss, valid_obj_loss, valid_cls_loss = 0, 0, 0, 0, 0
                valid_tqdm = tqdm.tqdm(valid_dataset, total=valid_dataset_length, ncols=160, desc=f'Valid epoch {epoch}/{epochs}', ascii=' =', colour='blue')
                for dist_batch_images, dist_batch_labels in valid_tqdm:
                    dist_batch_preds, total_loss, reg_loss, obj_loss, cls_loss = distributed_eval_step(dist_batch_images, dist_batch_labels)
                    update_eval_step(dist_batch_preds, dist_batch_labels)
                    mAP50, mAP = eval.compute_mAP()
                    
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
                
                if mAP > max_mAP:
                    max_mAP = mAP
                    save_model(model, epoch, mAP50, mAP, valid_loss, cfg['model']['best_checkpoint'])
                save_model(model, epoch, mAP50, mAP, valid_loss, cfg['model']['last_checkpoint'])

        dataloader.reset()

if __name__ == '__main__':
    main()
