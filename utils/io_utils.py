import tensorflow as tf

def write_model_info(checkpoints, epoch, mAP50, mAP, loss):
    with open(checkpoints + '.info', 'w') as f:
        text = f'epoch:{epoch}\n' +\
               f'mAP50:{mAP50}\n' +\
               f'mAP:{mAP}\n' +\
               f'total_loss:{loss[3]}\n' +\
               f'loc_loss:{loss[0]}\n' +\
               f'conf_loss:{loss[1]}\n' +\
               f'prob_loss:{loss[2]}\n'
        f.write(text)
        
def read_model_info(checkpoints):
    saved_parameter = {}
    with open(checkpoints + '.info', 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line[:-1].split(':')
            if key == 'epoch':
                saved_parameter[key] = int(value)
            else:
                saved_parameter[key] = float(value)
    return saved_parameter

def write_summary(writer, step, lr_or_mAP, loss, train=True):
    with writer.as_default():
        if train:
            tf.summary.scalar('lr', lr_or_mAP, step=step)
            tf.summary.scalar('train_loss/loc_loss', loss[0], step=step)
            tf.summary.scalar('train_loss/conf_loss', loss[1], step=step)
            tf.summary.scalar('train_loss/prob_loss', loss[2], step=step)
            tf.summary.scalar('train_loss/total_loss', loss[3], step=step)
            
        else:
            tf.summary.scalar("mAP50", lr_or_mAP[0], step=step)
            tf.summary.scalar("mAP", lr_or_mAP[1], step=step)
            tf.summary.scalar("validate_loss/loc_loss", loss[0], step=step)
            tf.summary.scalar("validate_loss/conf_loss", loss[1], step=step)
            tf.summary.scalar("validate_loss/prob_loss", loss[2], step=step)
            tf.summary.scalar("validate_loss/total_loss", loss[3], step=step)
        
    writer.flush()
    
def edit_config(pre_text, new_text):
    with open('config.py', 'r') as f:
        lines = f.readlines()
    for l in range(len(lines)):
        if pre_text in lines[l]:
            sp_line = lines[l].split(pre_text)
            lines[l] = sp_line[0] + str(new_text) + sp_line[1]
        elif 'CREATE_ANCHORS = True' in lines[l]:
            lines[l] = lines[l].replace('True', 'False')
    with open('config.py', 'w') as f:
        f.writelines(lines)
    
def write_eval(text, output_dir):
    path = output_dir + 'evaluation.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)