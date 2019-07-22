# tensorflow分布式

## 1.深度学习分布式的简介

现在流行的分布式的架构有PS，即参数服务器的，还有ring-allreduce，还有最新的字节跳动新推出的PS，先介绍一下这些ring-allreduce，也是最流行的。刚开始学习ring-allreduce时候也是很郁闷为什么取这个名字，其实是由reduce这个算法而来的，reduce是GPU基础算法，然后多个GPU变成了allreduce算法，再优化成ring-allreduce，可以说一步步演变而来，reduce算法可以看以下这个连接介绍：

<https://blog.csdn.net/abcjennifer/article/details/43528407>

介绍一下Reduce算法：

打个比方我们考虑一个task：1+2+3+4+…，想要得到和的结果，这其实就是一种reduce，所以reduce表示的是

- 数据：一个序列的数，如1,2,3,4...

- 操作符：满足两两输入，一个输出，比如这种可以写成((1+2)+3)+4…，其中加号这种操作符就满足这种性质，当然按位与也是，但是像a^b这种和减法都不是。

Reduce也包含串行和并行的(Serial implementation of Reduce和Parallel implementation of Reduce)，从字面上看，并行才有意义，节省时间，用图来介绍一下两种，如图：  
![reduce.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/reduce.png?raw=true)  
可以看到并行的reduce的话，相当于二叉树，而深度就是log<sub>2</sub>n，能够把原来n步的计算减少到log<sub>2</sub>n，更详细的可以参考博客里面的，还介绍了Scan和Histogram。
再来看看什么是allreduce算法，如图：

![allreduce0.png](https://github.com/wangjm12138/mnist_summarize/blob/master/markdown_pic/allreduce.png?raw=true)

也就是说reduce考虑是一组数组相加所提出的优化算法，allreduce就是对多组的数组进行并行的按下标相加的结果。process1到4就是四个数组，A<sub>1</sub>到A<sub>4</sub>，而最终每个process1到4都要变成B数组，而B数组的元素是对应是A<sub>1</sub>到A<sub>4</sub>对应的下标的之和，公式如下，B数组有4个元素，所以得算4个数组。
$$
B_i = A_{1,i}\quad Op\quad A_{2,i} \quad Op \quad ... A_{P,i}
$$
在深度学习分布式当中，也是这种情况，每个process可以看成每个GPU，而每个GPU上面有更新了相应自身的梯度，而每个GPU上面的梯度需要互相相加然后取均值，而相加的过程就像allreduce过程。常见的，如果把其中一个GPU当成master，然后其他GPU把梯度发送给master，相加后取完平均再依次分发给其他GPU，但是这样随着分布式GPU数目变多，master的GPU的通信量巨增，会造成网络拥塞，并且也会加重master的计算量(其实这部分跟PS架构很像，我自己本身认为PS架构也好，ring-allreduce也好，其实都是由用到allreduce的过程，只不过PS架构和ring-allreduce不一样在通信方面，至于计算这种累加的过程其实一样的)。所以当前最流行的是ring-allreduce，如图：
![allreduce1.png](https://github.com/wangjm12138/mnist_summarize/blob/master/markdown_pic/ring-allreduce1.png?raw=true)  
![allreduce2.png](https://github.com/wangjm12138/mnist_summarize/blob/master/markdown_pic/ring-allreduce2.png?raw=true)  
![allreduce3.png](https://github.com/wangjm12138/mnist_summarize/blob/master/markdown_pic/ring-allreduce3.png?raw=true)  
![allreduce4.png](https://github.com/wangjm12138/mnist_summarize/blob/master/markdown_pic/ring-allreduce4.png?raw=true)  
上面这个过程其实就是scatter-reduce的过程，最后，每个process之间在循环一次（不计算reduce）就可以将全部reduce后的值发送到每个process上，这过程叫allgather，为了更好展示ring-allreduce，用了网络上的图再次进行这两个scatter-reduce和allgather的过程。  
### The Scatter-Reduce
![newallreduce0.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter.png?raw=true) 
![newallreduce2.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter2.png?raw=true)
![newallreduce3.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter3.png?raw=true)
![newallreduce4.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter4.png?raw=true)
![newallreduce5.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter5.png?raw=true)
![newallreduce6.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter6.png?raw=true)
![newallreduce7.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter7.png?raw=true)
### The Allgather
![newallreduce8.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter8.png?raw=true)
![newallreduce9.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter9.png?raw=true)
![newallreduce10.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter10.png?raw=true)
![newallreduce11.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter11.png?raw=true)
![newallreduce12.png](https://github.com/wangjm12138/tensorflow_summarize/blob/master/markdown_pic/ring-allreduce_inter12.png?raw=true)
### ring-allreduce通信量的分析
#### 模型参数量
`D = 模型参数量在单精度下大小总和(MB)`  
以VGG-16为例，根据根据 https://dgschwend.github.io/netscope/#/preset/vgg-16 计算，整个模型共有138.36M个参数，按照单精度存储，则所有参数大小为
138.36M x 4bytes = 553.44MB
#### 通信速度
`S = 通信速度（MB/s)`  
当前服务器间通信使用千兆网卡，1000Mbit = 125Mbytes = 125MB
#### ring-allreduce通信量公式
N个GPU中的每一个都将发送和接收N-1次scatter-reduce，N-1次allgather。每次，GPU都会发送K / N值，其中K是数组中不同GPU上相加的值总数。因此，传输到每个GPU和从每个GPU传输的数据总量为  
$$
Data Transferred = 2(N-1)\frac{K}{N}
$$  
#### 并行效率-通信速率关系理论分析
`Ts = 单机单卡下，每个step的训练时间(s)`    
`Tp = 多卡情况下，每个step的训练时间(s)`  
说明：
在分布式训练中，我们通过使用多个GPU同时进行训练的方式来缩短训练时间  
每个step内，n个GPU执行的计算量是单GPU的n倍  
n个GPU只需单GPU下1/n的step数量即可获得近似的模型性能  

理想情况下，若Ts = Tp，则并行效率P = 100%，这是分布式训练性能的上确界，而事实上，分布式训练需要进行数据传输，记每个step数据传输时间为Tt，若忽略其他因素，则近似地：  
$$		
Tp = Ts + Tt
$$  
在本例中，并行效率
$$ 
P = \frac{Ts}{Tp} = \frac{Ts}{Ts+Tt} = \frac{1}{1+Tt/Ts}\qquad（1）
$$  
若并行效率要求为P, 取值范围[0,1]，则可得:
$$
Tt =\frac{(1-P)}{P}* Ts\qquad（2）
$$  
优化分布式训练的目标是最小化Tt    
根据以上分析，有如下：   
$$
Tt =\frac{2D(n-1)}{ns}
$$  
代入(1)，可得在固定通信速率下，并行效率为：  
$$
P =1/(\frac{2D(n-1)}{n * S * Ts})
$$  
计算在固定并行效率下需要的通信速率，可得：  
$$
S=\frac{2D(n-1)/n}{Tt}
$$   
带入(2)，有:
$$
S=\frac{2DP(n-1)}{n(1-P)Ts}\qquad（3）
$$  
#### 网卡性能需求分析
根据(1)式，代入目前测试参数（D = 553.36MB, S = 125MB/s， n=4，Ts ~= 0.55s)，可得理想并行效率（忽略通信损耗意外的其他因素）为：  
`P=1/(1+(2*553.36*3)/(4*125*0.55)) = 0.0765`  
执行每个step需要：  
`Tp=Ts/P=7.19s`  
与初步测试结果 Tp = 7.628基本相符  
根据（3）式，若要分布式效率达到90%，代入目前测试参数，则有  
`S=(2 * 553.36 * 3 * 0.9)/(4 * 0.1 * 0.55) ~= 13582 MB/s ~= 108660 Mbit/s`  
Important Note: 以上例子中，忽略了带宽使用率的问题，随着并行效率的增高，带宽使用率会逐步降低，在理想情况下，若并行效率为100%，则带宽使用率为0%，即完全不进行数据传输。因此，对通信速度的实际需求会比本文分析得高  
Note: 由于分析中忽略了网络损耗以外会导致并行效率降低的因素，所以实际需要的通信速度会更高;  
Note2: 上述数据仅适用于当前测试环境及模型，不同的设备及模型会有不同的D和Ts值;  
Note3：上述例子中，只使用了4个GPU，若GPU数量增多，通信需求将会逼近当前的1.33333倍(当前n=4，(n-1)/n=0.75，n逼近无限大时(n-1)/n=1, 1/0.75=1.333333)

