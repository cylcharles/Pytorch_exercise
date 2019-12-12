# Pytorch Exercise

Using Pytorch to do exercise.

## CNN with 3 Layers
建立3層卷積神經網路，使用10種類別的CIFAR10資料集訓練。

## Transfer Learning with ResNet18 on ImageNet
使用Pre-trained的ResNet進行遷移訓練，資料集使用10種類別的CIFAR10資料集訓練fine tune，基於Pre-trained的Resnet18 on ImageNet Dataset，進行Transfer
Learning只訓練5個Epochs，Accuracy就達到91%，相較於自建的3層CNN網路，訓練50個Epochs，準確度只有59%。
