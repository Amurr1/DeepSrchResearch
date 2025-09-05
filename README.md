# Alexis Murray's DeepSrch Research

This folder contains the explorations I have done with both EfficientNet and various ViT architectures for the purpose of classifying and/or performing weakly supervised segmentation.

*NOTE: All AUC metrics listed are for TS40 unless stated otherwise.*

### EffNetModifications

This is a small edit I made to our pre-existing EfficientNet, which pushed it from an AUC of ~0.997 to ~0.9993. I did this though replacing the existing LR scheduler with a ReduceLROnPlateau scheduler, which reduced the LR when the validation AUC does not improve for a preset number of epochs.

### TransformerExploration

This is where the majority of my efforts have taken me so far. I have tested 4 different models so far in direct training and studied less rigorusly a few more, but those are not included in this folder. 

Generally, I have found that EfficientFormer and EfficientViT were the best for classification, reaching ~0.997 AUC without extensive hyperparameter tuning. These two models are very similar in architecture as they are both hybrid transformers, which means they have an overall ResNet-like structure but add transformer blocks in the later half of the model. This combo is great for computational efficiency and classification, but not great for segmentation.

DINOv2 is a much more promising start for segmentation for a few reasons: 
- It has a CLS token, which is used for mapping classification outcomes to portions of the original image.
- It has better weakly supervised performance than other models.
- It has been improved through recent research on modified CLS tokens in these two papers:
   - Registers (https://arxiv.org/abs/2309.16588, 2024)
   - “Jumbo” CLS (https://arxiv.org/abs/2502.15021, 2025)

Unfortunately, I have not been about to get much higher than just above 0.99 AUC, and segmentation has proven difficult.

### ViT_From_Scratch

This is a project I am doing just to ensure my understanding of these models is correct. I do not intend on rebuilding the wheel and competing with models like DINOv2 at all, but I feel that this is a worthwhile educational exercise for me to undertake. Training is done on the MNIST dataset, not our working set of datasets with pretrained models.
