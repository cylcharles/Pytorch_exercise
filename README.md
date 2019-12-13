# Pytorch Exercise

Using Pytorch to do exercise.

## CNN with 3 Layers
建立3層卷積神經網路，使用10種類別的CIFAR10資料集訓練。

## Transfer Learning with ResNet18 on ImageNet
使用Pre-trained的ResNet進行遷移訓練，資料集使用10種類別的CIFAR10資料集訓練fine tune。
基於Pre-trained的Resnet18 on ImageNet Dataset，進行Transfer
Learning只訓練5個Epochs，Accuracy就達到91%，相較於自建的3層CNN網路，訓練50個Epochs，準確度只有59%。

## 補充資料
### Simple Backpropagation Conception
這邊用簡單小範例幫自己紀錄反向傳播的基本概念，先假設一個很簡單的網路，進行前向傳播 :
* 假設輸入是a、b和c，計算單元將a和b進行相加
```
x = a + b
```
* 而另一個計算單元是將b和c進行乘積 :
```
y = a * b
```
* 最後一層的計算單元則是將剛剛計算好的x函式和y函式進乘積 :
```
f = x * y
```
* 再來進行反向傳播解函數微分，對每一個計算單元(神經元)偏微分，首先由最終輸出f反推回x函數時，是對x進行偏微 :
```
df/dx =  y
```
* 再來是由f反推回y函數時，對y進行偏微 :
```
df/dy =  x
```
接著以此類堆計算所有函式的偏微分，完整示意圖如下圖，而有了這些值我們就可以計算a、b和c輸入出現變化時對輸出f的影響

![image](https://github.com/cylcharles/Pytorch_exercise/blob/master/img/example1.png)

接著用更實際的例子來計算，如果網路的輸入是橘子的數量和橘子單價，然後在加上稅金，則最後輸出就是我們購買橘子的總額。示意圖中，假設購買1顆橘子，1顆橘子單價100元，並且需加上10%的稅金，那最後總價就是220元。那我們就可以用反向傳播的概念來計算如果蘋果數量不一樣，對最後金額有多少影響;如果蘋果漲一元，對總金額有多少影響。
![image](https://github.com/cylcharles/Pytorch_exercise/blob/master/img/example2.png)

首先也是對每一個參數進行偏微，每條線偏微後的結果如紅字所示。
* 所以假設多買了1顆橘子，從最後輸出走回橘子數量的輸入，將經過的值相乘，結果是110元，意思是多買1顆橘子總金額會多110元 : 
```
1 x 1.1 x 100 = 110
```
* 所以假設橘子漲1元，從最後輸出走回橘子單價的輸入，將經過的值相乘，結果是2.2元，意思是1顆橘子漲1元總金額會多2.2元 : 
```
1 x 1.1 x 2 = 2.2
```
這樣就是一個反向傳播最基本的概念，用一個簡單的程式碼時做，程式碼放在Backpropagation.ipynb。
