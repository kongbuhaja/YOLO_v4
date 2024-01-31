import cv2, glob, sys
import numpy as np
from utils.bbox_utils import xywh_to_xyxy_np
import time

class Drawer():
    def __init__(self, cfg):
        self.colors = self.get_colors(cfg['data']['labels']['count'])
        self.data_name = cfg['data']['name']
        self.data_labels = cfg['data']['labels']['name']
        self.model_name = cfg['model']['name']

    def get_colors(self, num_classes):
        return [[np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)] for _ in range(num_classes)]

    def draw_labels(self, image, data, xywh=True):
        if xywh:
            data = xywh_to_xyxy_np(data, with_label=True)
        bboxes = data[..., :4].astype(np.int32)
        scores = np.ones_like(data[..., -1]) if data.shape[-1] ==5 else data[..., 4]
        classes = data[..., -1].astype(np.int32)
        
        for bbox, score, cls in zip(bboxes, scores, classes):
            color = self.colors[cls]
            cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
            cv2.putText(image, f'{self.data_labels[cls]}:{score:.3f}', (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color, 1)

class Painter(Drawer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dir = f"{cfg['eval']['dir']}/image"
        self.draw = cfg['eval']['draw']
        self.title = 'prediction' if self.draw%2==0 else 'gt_&_pred'
        self.count = len(glob.glob(f'{self.dir}/{self.title}_*.jpg'))
        
    def draw_image(self, image, labels, preds, xywh=True):
        image = image[..., ::-1]
        output = self.draw_labels(image.copy(), preds)
        if self.title != 'prediction':
            gt = self.draw_labels(image.copy(), labels, xywh=xywh)
            output = np.concatenate([gt, output], 1)
        
        self.write_image(output)

    def write_image(self, image):
        title = f'{self.title}_{self.count}.jpg'
        filename = f'{self.dir}/{title}'

        if self.draw//100:
            cv2.imshow(title, image)
            key = cv2.waitKey()
            if key == 27:
                cv2.destroyAllWindows()
                sys.exit()
            elif self.draw//110 or key == ord('s'):
                cv2.imwrite(filename, image)
            cv2.destroyWindow(title)
        else:
            cv2.imwrite(filename, image)
            self.count += 1
        
class Player(Drawer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.total_frames = 0
        self.video = cfg['eval']['video']
        self.cap = cv2.VideoCapture(self.video)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.title = f"cam{self.video}" if isinstance(self.video, int) else self.video.split('/')[-1]
        self.dir = f"{cfg['eval']['dir']}/video/{self.title}"
        self.draw = cfg['eval']['draw']
        self.title = 'prediction' if self.draw%2==0 else 'gt_&_pred'
        self.count = len(glob.glob(f'{self.dir}_*.mp4'))
        self.writer = cv2.VideoWriter(f'{self.dir}_{self.count}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (self.width, self.height))
    
        print(f'video path:{self.video}')
        print(f'video size:{self.width} x {self.height}')
        print(f'video fps:{self.fps}')

    def __call__(self, frame, preds):
        self.cur_time = time.time()
        sec = self.cur_time - self.prev_time
        fps = 1 / sec
        self.prev_time = self.cur_time
        text = f'Inference time: {sec:.3f}, fps: {fps:.1f}'

        self.draw_labels(frame, preds)
        self.put_text(frame, text)
        self.write(frame)

        self.total_frames += 1

    def init_time(self):
        self.start_time = time.time()
        self.prev_time = self.start_time
    
    def read(self):
        return self.cap.read()

    def put_text(self, frame, text=''):
        cv2.putText(frame, text, (10,10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255), 1)

    def show(self, frame):
        cv2.imshow(self.title, frame)
    
    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.end_time = time.time()
        sec = self.end_time - self.start_time
        avg_fps = self.total_frames / sec
        print(f'Average inferrence time:{1/avg_fps:.3f}, Average fps:{avg_fps:.3f}')

        self.writer.release()
        self.cap.release()
        cv2.destroyAllWindows()