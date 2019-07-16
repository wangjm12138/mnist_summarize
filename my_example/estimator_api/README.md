# tensorflow分布式

## 1.深度学习分布式的简介

现在流行的分布式的架构有PS，即参数服务器的，还有ring-allreduce，还有最新的字节跳动新推出的PS，先介绍一下这些ring-allreduce，也是最流行的。刚开始学习ring-allreduce时候也是很郁闷为什么取这个名字，其实是由reduce这个算法而来的，reduce是GPU基础算法，然后多个GPU变成了allreduce算法，再优化成ring-allreduce，可以说一步步演变而来，reduce算法可以看以下这个连接介绍：

<https://blog.csdn.net/abcjennifer/article/details/43528407>

介绍一下Reduce算法：

打个比方我们考虑一个task：1+2+3+4+…，想要得到和的结果，这其实就是一种reduce，所以reduce表示的是

- 数据：一个序列的数，如1,2,3,4...

- 操作符：满足两两输入，一个输出，比如这种可以写成((1+2)+3)+4…，其中加号这种操作符就满足这种性质，当然按位与也是，但是像a^b这种和减法都不是。

Reduce也包含串行和并行的(Serial implementation of Reduce和Parallel implementation of Reduce)，从字面上看，并行才有意义，节省时间，用图来介绍一下两种，如图：  
![reduce.png](https://github.com/wangjm12138/mnist_summarize/blob/master/markdown_pic/redece.png?raw=true)

再来看看什么是allreduce算法，如图：

![allreduce.png](https://github.com/wangjm12138/mnist_summarize/blob/master/markdown_pic/allreduce.png?raw=true)

也就是说reduce考虑是一组数组相加所提出的优化算法，allreduce就是对多组的数组进行并行的按下标相加的结果。process1到4就是四个数组，A<sub>1</sub>到A<sub>4</sub>，而最终每个process1到4都要变成B数组，而B数组的元素是对应是A<sub>1</sub>到A<sub>4</sub>对应的下标的之和，公式如下，B数组有4个元素，所以得算4个数组。
$$
B_i = A_{1,i}\quad Op\quad A_{2,i} \quad Op \quad ... A_{P,i}
$$
在深度学习分布式当中，也是这种情况，每个process可以看成每个GPU，而每个GPU上面有更新了相应自身的梯度，而每个GPU上面的梯度需要互相相加然后取均值，而相加的过程就像allreduce过程。常见的，如果把其中一个GPU当成master，然后其他GPU把梯度发送给master，相加后取完平均再依次分发给其他GPU，但是这样随着分布式GPU数目变多，master的GPU的通信量巨增，会造成网络拥塞，并且也会加重master的计算量(其实这部分跟PS架构很像，我自己本身认为PS架构也好，ring-allreduce也好，其实都是由用到allreduce的过程，只不过PS架构和ring-allreduce不一样在通信方面，至于计算这种累加的过程其实一样的)。所以当前最流行的是ring-allreduce，如图：

