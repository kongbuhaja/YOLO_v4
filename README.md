# YOLOv4  

## Modules  
### BoF for backbone  
- [ ] CutMix   
- [ ] Mosaic 
- [ ] DropBlock  
- [x] Class label smoothing  
### BoS for backbone  
- [x] Mish activation  
- [x] Cross-stage partial connections(CSP)  
- [x] Multi-input weighted residual connections(MiWRC)  
### BoF for detector  
- [x] CIoU-loss  
- [ ] CmBN  
- [ ] DropBlock
- [ ] Mosaic  
- [x] Self-Adversarial Training(SAT)  
- [x] Eliminate grid sensityivity  
- [x] multiple anchors for single ground truth
- [x] Cosine annealing scheduler
### Bos for detector  
- [x] Mish activation  
- [x] Spatial Pyramid Pooling(SPP)  
- [x] Modified Spatical Attention Module(SAM)
- [x] Modified Path Aggregation Network(PAN)
- [x] DIoU_NMS

## Performance  
### MS COCO 
| Model | Test Size | AP | AP50 |  
| ------------- | ------ | ------ | ------ |
| VOLOv4_tiny  | - | - | - |
| VOLOv4  | - | - | - |
| VOLOv4_csp  | - | - | - |
| VOLOv4_P5  | - | - | - |
| VOLOv4_P6  | - | - | - |
| VOLOv4_P7  | - | - | - |
| YOLOv3_tiny  | - | - | - |
| VOLOv3  | - | - | - |
| VOLOv2_tiny  | - | - | - |
| VOLOv2  | - | - | - |

### VOC Pascal 
| Model | Test Size | AP | AP50 |  
| ------------- | ------ | ------| ------|
| VOLOv4_tiny  | - | - | - |
| VOLOv4  | - | - | - |
| VOLOv4_csp  | - | - | - |
| VOLOv4_P5  | - | - | - |
| VOLOv4_P6  | - | - | - |
| VOLOv4_P7  | - | - | - |
| YOLOv3_tiny  | - | - | - |
| VOLOv3  | - | - | - |
| VOLOv2_tiny  | - | - | - |
| VOLOv2  | - | - | - |

### Custom dataset  
| Model | Test Size | AP | AP50 |  
| ------------- | ------ | ------ | ------ |
| VOLOv4_tiny  | - | - | - |
| VOLOv4  | - | 70.5% | 86.0% |
| VOLOv4_csp  | - | 70.6% | 84.6% |
| VOLOv4_P5  | - | - | - |
| VOLOv4_P6  | - | - | - |
| VOLOv4_P7  | - | - | - |
| YOLOv3_tiny  | - | - | - |
| VOLOv3  | - | - | - |
| VOLOv2_tiny  | - | - | - |
| VOLOv2  | - | - | - |
