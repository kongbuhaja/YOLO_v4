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
- [ ] Cosine annealing scheduler
### Bos for detector  
- [x] Mish activation  
- [x] Spatial Pyramid Pooling(SPP)  
- [x] Modified Spatical Attention Module(SAM)
- [x] Modified Path Aggregation Network(PAN)
- [x] DIoU_NMS
