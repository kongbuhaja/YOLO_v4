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

## Result   
### COCO  
|Model|Image Size|mAP|  
|-----|----------|-----|
|YOLOv4_P7|1536x1536|-|  
|YOLOv4_P6|1280x1280|-|  
|YOLOv4_P5|896x896|-|  
|YOLOv4_csp|512x512|0.1013| 
|YOLOv4|512x512|0.092| 
|YOLOv3|512x512|0.025|  
|YOLOv4_tiny|512x512|  
|YOLOv3_tiny|416x416|  

### VOC  
|Model|Image Size|mAP|  
|-----|----------|-----|
|YOLOv4_P7|1536x1536|-|  
|YOLOv4_P6|1280x1280|-| 
|YOLOv4_P5|896x896|-|  
|YOLOv4_csp|512x512|0.1013|  
|YOLOv4|512x512|0.092|  
|YOLOv3|512x512|0.025|  
|YOLOv4_tiny|512x512|  
|YOLOv3_tiny|416x416|  

### Custom  
|Model|Image Size|mAP|  
|-----|----------|-----|
|YOLOv4_P7|1536x1536|-|  
|YOLOv4_P6|1280x1280|0.3877|  
|YOLOv4_P5|896x896|0.4994|  
|YOLOv4_csp|512x512|0.4036|  
|YOLOv4|512x512|0.3583|  
|YOLOv3|512x512|0.3924|  
|YOLOv4_tiny|512x512|0.3941|  
|YOLOv3_tiny|416x416|0.4030|  
