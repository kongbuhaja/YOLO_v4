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

SPPBlock, CSPSPPBlock, PlainBlock, UpsampleBlock, ResidualBlock 완료
Downsample까지 만들고 FPN, PAN 이상없는지 대충확인하고
CSPBlock 활용처 확인