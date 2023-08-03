import cv2, glob, sys
import numpy as np

def draw_labels(image, preds, labels, xywh=True):
    bboxes = preds[..., :4].astype(np.int32)
    scores = preds[..., 4]
    classes = preds[..., 5].astype(np.int32)
    if xywh:
        xy1 = bboxes[..., :2] - 0.5 * bboxes[..., 2:4]
        xy2 = bboxes[..., :2] + 0.5 * bboxes[..., 2:4]
        bboxes = np.concatenate([xy1, xy2], -1).astype(np.int32)
    for bbox, score, cls in zip(bboxes, scores, classes):
        if np.sum(bbox)==0:
            break;
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
        cv2.putText(image, f'{labels[cls]}:{score:.3f}', (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color, 1)
    return image

def draw_image(image, input_size, output_dir, draw=True):
    if image.shape[1] != input_size:
        title = 'truth_and_pred'
    else:
        title = 'prediction'
    
    filename = output_dir + 'image/' + title
    filename += '_' + str(len(glob.glob(filename + '*.jpg')))
    
    if draw:
        cv2.imwrite(filename + '.jpg', image)
    else:
        cv2.imshow(title, image)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit()
        elif key == ord('s'):
            cv2.imwrite(filename + '.jpg', image)
        cv2.destroyWindow(title)
    
    