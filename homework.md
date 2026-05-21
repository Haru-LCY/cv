Assignment 3: YOLO Training from Scratch
Objective
Implement a YOLO object detector from scratch and train it on a face detection dataset. YOLO is a classic real-time object detection framework that formulates detection as a single regression problem. ([arXiv][1])
Part 1 — Build the YOLO Codebase
Implement the basic components of a YOLO detector (or other YOLO-series variants) using PyTorch; you may reference existing implementations for guidance, and the codebase should be trainable with components including dataset loader, backbone network, detection head, loss function, NMS/inference pipeline, and training script.
Part 2 — Train on a Face Dataset
Train and evaluate the model on a classic face detection dataset such as:
• FDDB
• MALF
• WIDER FACE
If computational resources are limited, students may sample a subset of the dataset for training.
Required results:
• Training loss curve
• mAP metric (mAP@.5, mAP@.9, mAP@[.5:.95] )
• Precision–Recall curves under an IoU threshold of 0.7.

Part 3 — Experiments and Discussion (Optional)
Students may optionally explore different modules or hyperparameters and briefly discuss their effects on performance.
Submission
Submit:
1. Source code
2. An English report including:
• Experimental results
• Curves/visualizations
• Brief discussion and conclusion

Note: Trained model weights are not required.

[1]: https://arxiv.org/abs/1506.02640  "You Only Look Once: Unified, Real-Time Object Detection"
