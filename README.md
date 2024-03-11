# YOLOv4  

## Implementation Modules  
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
| ------------- | ------ | ------| ------ |
| **VOLOv4_tiny**  | 416x416 | - | - |
| **VOLOv4**  | 512x512 | - | - |
| **VOLOv4_csp**  | 512x512 | - | - |
| **VOLOv4_P5**  | 896x896 | - | - |
| **VOLOv4_P6**  | 1280x1280 | - | - |
| **VOLOv4_P7**  | 1536x1536 | - | - |
| **YOLOv3_tiny**  | 416x416 | - | - |
| **VOLOv3**  | 512x512 | - | - |
| **VOLOv2_tiny**  | 416x416 | - | - |
| **VOLOv2**  | 416x416 | - | - |

### VOC Pascal 
| Model | Test Size | AP | AP50 |  
| ------------- | ------ | ------| ------ |
| **VOLOv4_tiny**  | 416x416 | - | - |
| **VOLOv4**  | 512x512 | - | - |
| **VOLOv4_csp**  | 512x512 | - | - |
| **VOLOv4_P5**  | 896x896 | - | - |
| **VOLOv4_P6**  | 1280x1280 | - | - |
| **VOLOv4_P7**  | 1536x1536 | - | - |
| **YOLOv3_tiny**  | 416x416 | - | - |
| **VOLOv3**  | 512x512 | - | - |
| **VOLOv2_tiny**  | 416x416 | - | - |
| **VOLOv2**  | 416x416 | - | - |

### Custom dataset  
| Model | Test Size | AP | AP50 |  
| ------------- | ------ | ------ | ------ |
| **VOLOv4_tiny**  | 416x416 | - | - |
| **VOLOv4**  | 512x512 | 70.5% | 86.0% |
| **VOLOv4_csp**  | 512x512 | 70.6% | 84.6% |
| **VOLOv4_P5**  | 896x896 | 62.2% | 78.5% |
| **VOLOv4_P6**  | 1280x1280 | - | - |
| **VOLOv4_P7**  | 1536x1536 | - | - |
| **YOLOv3_tiny**  | 416x416 | - | - |
| **VOLOv3**  | 512x512 | - | - |
| **VOLOv2_tiny**  | 416x416 | - | - |
| **VOLOv2**  | 416x416 | - | - |
