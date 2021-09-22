---
layout: post
title:  "CornerNet: Detecting Objects as Paired Keypoints(代码未添加)"
date:   2021-08-24 16:00:00 +0800
tags: [deep learning, algorithm, detection]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/CornerNet>

CornerNet论文：

<https://arxiv.org/abs/1808.01244>

CornerNet-Lite论文：

<https://arxiv.org/abs/1904.08900>

官方CornerNet-Lite的pytorch代码：

<https://github.com/princeton-vl/CornerNet-Lite>

说明：本文为cornernet的理解。但代码看的是CornerNet-Lite，不太匹配。。。


# P1. 简介
该文提出CornerNet，一个不需要anchor的一阶段目标检测算法。将目标建模为目标框的左上角和右下角的角点，并使用CNN分别预测当前类别所有目标左上角和右下角角点的热图，以及每个角点的特征向量（embedding vector）。特征向量用于对相同目标的角点分组，训练网络来预测相同的特征向量，使得相同目标的两个角点的特征向量间的距离更小。该算法简化了网络的输出，不再需要设计anchor，整体框图如图1所示。

![1](/assets/post/2021-08-24-CornerNet/1cornernet.png)
_图1 cornernet_

该文还提出了corner pooling，一个新的池化层，帮助CNN更好的定位边界框的角点。边界框的角点通常是在目标外面（才能把目标框起来）。此时无法根据局部信息定位角点。因而为了确定某个位置是否是左上角的角点，需要水平向右看物体的最高边缘，同时垂直向下看物体的最左侧边缘。这促使我们提出了corner pooling层：其包含两个特征图，在每个位置其在第一个特征图取上取当前点到最右侧所有特征的最大值（max-pool），同时在第二个特征图上取当前点到最下侧所有特征的最大值（max-pool），最终把这两个池化结果相加。如图2所示。

![2](/assets/post/2021-08-24-CornerNet/2cornernetpooling.png)
_图2 cornernetpooling_

为何使用角点比使用边界框中心或候选框要好，作者给出了两个猜测的原因：① 由于目标框需要依赖目标的4个边界，因而目标框的中心比较难于定位；而定位角点只需要2个边界，因而更容易定位，corner pooling更是如此，其编码了一些角点的先验知识。② 角点更高效的提供了密集离散化边界框的方式：只需要
$$O\left( wh \right)$$
 个角点就能代表
$$O\left( { {w}^{2}}{ {h}^{2}} \right)$$
个可能的anchor。

# P2. CornerNet

## P2.1 简介

该算法将目标建模为目标框的左上角和右下角的角点，并使用CNN分别预测当前类别所有目标左上角和右下角角点的热图，以及每个角点的特征向量（embedding vector），使得相同目标的两个角点的特征向量间的距离更小。为了生成更准确的边界框，网络也会预测偏移，来轻微的调整角点的位置。图3是CornerNet的框图。使用hourglass作为骨干网络。骨干网络之后接两个预测模块。一个预测模块用于预测左上角角点，另一个预测右下角角点。每个模块都有相应的角点池化模层，用于对hourglass的特征进行池化，之后得到预测热图、特征向量和预测偏移。本文不使用不同尺度的特征，只使用hourglass输出特征。

![3](/assets/post/2021-08-24-CornerNet/3overview.png)
_图3 overview of cornernet_

## P2.2 角点检测

本文预测2组热图，一组用于左上角角点，一组用于右下角角点。每组热图有C个通道，且无背景通道，热图宽高为H\*W，C为目标检测类别的个数。每个通道是二值掩膜，代表当前位置是某个类别的角点。

每个角点有一个GT位置作为正位置，所有其他位置都为负位置。在训练阶段不是等价的惩罚负样本位置，而是减少了对正位置半径内负位置的惩罚。原因是如果一对错误的角点靠近各自的GT位置，仍然可以产生一个与GT框充分重叠的框，如图4所示。本文根据对象的大小确定半径，确保半径内的一对点对应的框和GT框具有至少t IoU（实验中t=0.3）。得到半径后，通过未归一化的2D高斯核
$${ {e}^{-\frac{ { {x}^{2}}+{ {y}^{2}}}{2{ {\sigma }^{2}}}}}$$
 来进行惩罚，高斯核的中心为GT位置，
 $$\sigma $$
 为半径的1/3。

![4](/assets/post/2021-08-24-CornerNet/4gt.png)
_图4_

令
$${ {p}_{cij}}$$
为预测热图中类别c、位置
$$\left( i,j \right)$$
处的得分，
$${ {y}_{cij}}$$
为通过未归一化的高斯核得到的GT热图，本文设计focal loss的变种：

$${ {L}_{\det }}=\frac{-1}{N}\sum\limits_{c=1}^{C}{\sum\limits_{i=1}^{H}{\sum\limits_{j=1}^{W}{\left\{ \begin{matrix}
   { {\left( 1-{ {p}_{cij}} \right)}^{\alpha }}\log \left( { {p}_{cij}} \right) & if\text{ }{ {y}_{cij}}=1  \\
   { {\left( 1-{ {y}_{cij}} \right)}^{\beta }}{ {\left( { {p}_{cij}} \right)}^{\alpha }}\log \left( 1-{ {p}_{cij}} \right) & otherwise  \\
\end{matrix} \right.}}} \tag{1}$$

其中N为图像中目标的数量，
$$\alpha $$
和
$$\beta $$
为控制每个点分布的超参（实验中设置
$$\alpha =2$$
，
$$\beta =4$$
）。

$$1-{ {y}_{cij}}$$
能够降低GT位置附近的惩罚。

很多网络使用下采样获取全局信息，同时降低显存需求，但会导致网络输出尺寸小于图像尺寸。因而图像上
$$\left( x,y \right)$$
位置会映射到特征图上的
$$\left( \left\lfloor \frac{x}{n} \right\rfloor ,\left\lfloor \frac{y}{n} \right\rfloor  \right)$$
，其中n为下采样率。当从特征图重新映射到输入图像时，会导致精度损失，从而严重影响小边界框和他们GT框的IoU。为了解决这个问题，本文预测位置偏移，在将位置映射回输入尺寸之前轻微调整角点的位置。

$${ {o}_{k}}=\left( \frac{ { {x}_{k}}}{n}-\left\lfloor \frac{ { {x}_{k}}}{n} \right\rfloor ,\frac{ { {y}_{k}}}{n}-\left\lfloor \frac{ { {y}_{k}}}{n} \right\rfloor  \right) \tag{2}$$

其中
$${ {o}_{k}}$$
为偏移，
$${ {x}_{k}}$$
和
$${ {y}_{k}}$$
为角点k的x和y坐标。实际中会预测所有类别共享的左上角偏移，及所有类别共享的右下角偏移。使用smooth L1 loss训练偏移：

$${ {L}_{off}}=\frac{1}{N}\sum\limits_{k=1}^{N}{\text{SmoothL1Loss}\left( { {o}_{k}},{ { {\hat{o}}}_{k}} \right)} \tag{3}$$

smooth L1 loss如下<https://arxiv.org/abs/1504.08083>：

$$\text{smoot}{ {\text{h}}_{\text{L1}}}\left( x \right)=\left\{ \begin{matrix}
   0.5{ {x}^{2}} & if\text{ }\left| x \right|<1  \\
   \left| x \right|-0.5 & otherwise  \\
\end{matrix} \right. \tag{3.1}$$ 

说明：公式3.1中x等于公式3中
$${ {o}_{k}}-{ {\hat{o}}_{k}}$$


## P2.3 角点分组

一张图像中可能出现多个类别的目标，可能检测到多个左上角和右下角角点，从而需要确定一组左上角和右下角角点属于同一个边界框。本文受<https://arxiv.org/abs/1611.05424>中多人姿态估计算法启发。该文检测所有人的关节点，并生成每个关节点的特征向量（embedding），然后根据特征向量间的距离对关节点分组。该算法也能用于本文中。网络对每个检测到的角点预测特征向量，该向量代表预测到的左上角角点和右下角角点是否来自同一个框（相同目标），同一个框的embedding向量的距离更小。之后根据左上角和右下角角点的特征向量的距离对角点分组。特征向量的实际值不重要，因为使用的是特征之间的距离来对角点分组。

本文使用1维的特征向量。令
$${ {e}_{ { {t}_{k}}}}$$
为目标k左上角角点的特征，
$${ {e}_{ { {b}_{k}}}}$$
为目标k右下角角点的特征，本文使用pull loss训练网络来聚集角点，使用push loss来分散角点：

$${ {L}_{pull}}=\frac{1}{N}\sum\limits_{k=1}^{N}{\left[ { {\left( { {e}_{ { {t}_{k}}}}-{ {e}_{k}} \right)}^{2}}+{ {\left( { {e}_{ { {b}_{k}}}}-{ {e}_{k}} \right)}^{2}} \right]} \tag{4}$$

$${ {L}_{push}}=\frac{1}{N\left( N-1 \right)}\sum\limits_{k=1}^{N}{\sum\limits_{\begin{smallmatrix} 
 j=1 \\ 
 j\ne k 
\end{smallmatrix}}^{N}{\max \left( 0,\Delta -\left| { {e}_{k}}-{ {e}_{j}} \right| \right)}} \tag{5}$$

其中
$${ {e}_{k}}$$
为
$${ {e}_{ { {t}_{k}}}}$$
和
$${ {e}_{ { {b}_{k}}}}$$
的平均值，即中心，
$$\Delta =1$$
，push loss是左上角和右下角角点的中心之间相互比较。和训练偏移的损失类似，本文只对GT位置的角点计算pull loss和push loss。

## P2.4 角点池化（Corner Pooling）

角点通常没有局部视觉信息。因而需要分别需要水平向右看物体的最高边缘、垂直向下看物体的最左侧边缘，才能确定左上角的角点。因而该文提出角点池化（Corner Pooling），来给角点编码先验知识，更好的定位角点。

假定需要确定位置
$$\left( i,j \right)$$
的像素是否是左上角点。令
$${ {f}_{t}}$$
和
$${ {f}_{l}}$$
分别为左上角点池化层的输入，
$${ {f}_{ { {t}_{ij}}}}$$
和
$${ {f}_{ { {l}_{ij}}}}$$
分别为
$${ {f}_{t}}$$
和
$${ {f}_{l}}$$
中位置
$$\left( i,j \right)$$
处的向量。对于H\*W的特征图，角点池化层首先最大池化（max-pool）
$${ {f}_{t}}$$
中所有的
$$\left( i,j \right)$$
和
$$\left( i,H \right)$$
的特征到特征向量
$${ {t}_{ij}}$$
中，然后最大池化（max-pool）
$${ {f}_{l}}$$
中所有的
$$\left( i,j \right)$$
和
$$\left( W,j \right)$$
的特征到特征向量
$${ {l}_{ij}}$$
中。最后将
$${ {t}_{ij}}$$
和
$${ {l}_{ij}}$$
的结果相加：

$${ {t}_{ij}}=\left\{ \begin{matrix}
   \max \left( { {f}_{ { {t}_{ij}}}},{ {t}_{\left( i+1 \right)j}} \right) & if\text{ }i<H  \\
   { {f}_{ { {t}_{Hj}}}} & otherwise  \\
\end{matrix} \right. \tag{6}$$

$${ {l}_{ij}}=\left\{ \begin{matrix}
   \max \left( { {f}_{ { {l}_{ij}}}},{ {t}_{i\left( j+1 \right)}} \right) & if\text{ }j<W  \\
   { {f}_{ { {t}_{iW}}}} & otherwise  \\
\end{matrix} \right. \tag{7}$$

公式6中，当前点的值是当前点右侧所有值的最大的。i从H开始\-\-，进行比较。公式7中，当前点的值是当前点下方所有值的最大的。j从W开始\-\-，进行比较。
$${ {t}_{ij}}$$
和
$${ {l}_{ij}}$$
可以使用动态规划高效计算，如图5所示。最终
$${ {t}_{ij}}\text{+}{ {l}_{ij}}$$
得到结果。

![5](/assets/post/2021-08-24-CornerNet/5tlcornerpooling.png)
_图5_

右下角池化层通过类似方式定义。其最大池化
$$\left( 0,j \right)$$
和
$$\left( i,j \right)$$
的特征向量，以及
$$\left( i,0 \right)$$
和
$$\left( i,j \right)$$
的特征向量，并把相应结果相加。角点池化层用在预测模块，来预测热图、特征向量和目标偏移。

预测模块如图6所示。第一部分为修改的残差模块，此处将第一个3\*3卷积替换2个具有128通道的3\*3卷积模块模块，用来处理骨干网络的特征，卷积模块之后为角点池化层。按照残差模块的设计，我们将池化后的特征输入一个具有256个通道的3\*3 Conv-BN层，并添加shortcut支路。修改后的残差模块后面接具有256通道的3\*3卷积模块，3个Conv-ReLU-Conv层，来预测热图、特征向量和目标偏移。

![6](/assets/post/2021-08-24-CornerNet/6predictionmodule.png)
_图6 predictionmodule_

## P2.5 Hourglass网络

CornerNet的骨干网络为hourglass网络。其为全卷积网络，包含至少一个hourglass模块。hourglass模块先通过一系列的conv和max pooling来下采样输入特征，而后通过一系列上采样和卷积层上采样特征到原始分辨率。使用hourglass时，由于上采样丢失了细节信息，因而将输入层和上采样层使用skip层相连。
本文使用的hourglass网络包含2个hourglass模块，但对hourglass模块做了一些修改。我们没有使用max pooling，而是使用stride=2来降低特征图的分辨率。本文降低特征图的分辨率5次，同时特征通道数依次变为(256, 384, 384, 384, 512)。而后使用2个残差模块和一个最近邻上采样模块来上采样特征。每个skip连接也包括2个残差模块。在hourglass模块的中间有4个具有512个通道的残差模块。在hourglass模块之前，使用stride=2、128个通道的7\*7卷积，以及stride=2、256个通道的残差块将图像分辨率降低4倍。

本文训练阶段也使用中间监督。不过我们发现中间预测会降低网络的性能，因而没有将中间预测添加到网络中。我们在第一个hourglass模块的输入和输出都使用1\*1 conv bn，然后使用逐元素相加并加上ReLU和256通道的残差模块，作为第二个hourglass模块的输入。hourglass网络的层数为104。我们仅使用网络最后一层的特征进行预测。

## P2.6 损失

该文使用Adam训练网络。总体的损失为：

$$L={ {L}_{det}}+\alpha { {L}_{pull}}+\beta { {L}_{push}}+\gamma { {L}_{off}} \tag{8}$$

其中
$$\alpha =0.1$$
、
$$\beta =0.1$$
、
$$\gamma =1$$
分别是pull loss，push loss、偏移损失的权重。
$$\alpha $$
和
$$\beta $$
大于等于1时，模型性能变差。


# P3.代码

说明：相关代码有一定注释；调用系统函数的代码，基本无注释。

## P3.1 训练train.py

用于初始化pytorch分布式训练或者单机训练。代码如下：

<details>

```python
def train(training_dbs, validation_db, system_config, model, args):
    # reading arguments from command
    start_iter  = args.start_iter
    distributed = args.distributed
    world_size  = args.world_size
    initialize  = args.initialize
    gpu         = args.gpu
    rank        = args.rank

    # reading arguments from json file
    batch_size       = system_config.batch_size
    learning_rate    = system_config.learning_rate
    max_iteration    = system_config.max_iter
    pretrained_model = system_config.pretrain
    stepsize         = system_config.stepsize
    snapshot         = system_config.snapshot
    val_iter         = system_config.val_iter
    display          = system_config.display
    decay_rate       = system_config.decay_rate
    stepsize         = system_config.stepsize

    print("Process {}: building model...".format(rank))
    nnet = NetworkFactory(system_config, model, distributed=distributed, gpu=gpu)
    if initialize:
        nnet.save_params(0)
        exit(0)

    # queues storing data for training
    training_queue   = Queue(system_config.prefetch_size)
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_config.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # allocating resources for parallel reading
    training_tasks = init_parallel_jobs(system_config, training_dbs, training_queue, data_sampling_func, True)
    if val_iter:
        validation_tasks = init_parallel_jobs(system_config, [validation_db], validation_queue, data_sampling_func, False)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("Process {}: loading from pretrained model".format(rank))
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        nnet.load_params(start_iter)
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        nnet.set_lr(learning_rate)
        print("Process {}: training starts from iteration {} with learning_rate {}".format(rank, start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    if rank == 0:
        print("training start...")
    nnet.cuda()
    nnet.train_mode()   # 训练模式
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss = nnet.train(**training)

            if display and iteration % display == 0:
                print("Process {}: training loss at iteration {}: {}".format(rank, iteration, training_loss.item()))
            del training_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()  # eval模式
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("Process {}: validation loss at iteration {}: {}".format(rank, iteration, validation_loss.item()))
                nnet.train_mode()

            if iteration % snapshot == 0 and rank == 0:
                nnet.save_params(iteration)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)   # 更新学习率

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    terminate_tasks(training_tasks)
    terminate_tasks(validation_tasks)

def main(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  # 设置分布式训练

    rank = args.rank

    cfg_file = os.path.join("./configs", args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)

    config["system"]["snapshot_name"] = args.cfg_file
    system_config = SystemConfig().update_config(config["system"])

    model_file  = "core.models.{}".format(args.cfg_file)  # 从core.models.CornerNet载入模型
    model_file  = importlib.import_module(model_file)
    model       = model_file.model()   # 得到实际的模型，如core.models.CornerNet中的model

    train_split = system_config.train_split   # trainval
    val_split   = system_config.val_split     # minival

    print("Process {}: loading all datasets...".format(rank))   # rank为分布式训练的参数
    dataset = system_config.dataset   # COCO
    workers = args.workers   # 4
    print("Process {}: using {} workers".format(rank, workers))
    training_dbs = [datasets[dataset](config["db"], split=train_split, sys_config=system_config) for _ in range(workers)]  # datasets[dataset]=COCO，得到训练的
    validation_db = datasets[dataset](config["db"], split=val_split, sys_config=system_config)

    if rank == 0:
        print("system config...")
        pprint.pprint(system_config.full)

        print("db config...")
        pprint.pprint(training_dbs[0].configs)

        print("len of db: {}".format(len(training_dbs[0].db_inds)))
        print("distributed: {}".format(args.distributed))

    train(training_dbs, validation_db, system_config, model, args)

if __name__ == "__main__":
    args = parse_args()

    distributed = args.distributed
    world_size  = args.world_size

    if distributed and world_size < 0:
        raise ValueError("world size must be greater than 0 in distributed training")

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(None, ngpus_per_node, args)
```
</details>


## P3.2 网络core.models.CornerNet

位于core/models/CornerNet.py

### P3.2.1 CornerNet

<details>

```python
def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

class model(hg_net):   # cornernet网络结构，继承自hg_net
    def _pred_mod(self, dim):  # conv+bn+relu + 1*1conv
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        stacks  = 2
        pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),    # conv+bn+relu
            residual(128, 256, stride=2)         # 残差连接模块 (conv+bn + (conv+bn+relu+conv+bn)) + relu
        )
        hg_mods = nn.ModuleList([hg_module(5, [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4],
                                           make_pool_layer=make_pool_layer, make_hg_layer=make_hg_layer) for _ in range(stacks)])   # 2个hg_module模块

        cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])  # 2个
        inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])    # 1个
        cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])     # 1个
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])     # 1个
        
        # hg如下
        # 
        #                    out[0]                                                   out[0]
        #                      ↑
        #       hg_mods[0] → cnvs[0] → cnvs_[0]
        #     ↗                                ↘                                      ↑
        # pre                                      + → relu → inters[0] → hg_mods[1] → cnvs[1]
        #    ↘             inters_[0]          ↗  
        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_) 

        # corner_pool如下
        # pool1: TopPool or BottomPool
        # pool2: LeftPool or RightPool
        # 
        #         conv+bn+relu → pool1 
        #      ↗                      ↘ 
        #    x →  conv+bn+relu → pool2 →  + → conv → bn ↘
        #      ↘                                          + → relu → conv+bn+relu
        #         conv → bn  -------------------------→ ↗

        tl_modules = nn.ModuleList([corner_pool(256, TopPool, LeftPool) for _ in range(stacks)])       # 左上角池化
        br_modules = nn.ModuleList([corner_pool(256, BottomPool, RightPool) for _ in range(stacks)])   # 右下角池化

        tl_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])   # 预测目标左上角热图的分支  # conv+bn+relu + 1*1conv
        br_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])   # 预测目标右下角热图的分支  # conv+bn+relu + 1*1conv
        for tl_heat, br_heat in zip(tl_heats, br_heats):
            torch.nn.init.constant_(tl_heat[-1].bias, -2.19)   # 初始化bias
            torch.nn.init.constant_(br_heat[-1].bias, -2.19)

        tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])   # 预测左上角目标embedding向量的分支  # conv+bn+relu + 1*1conv
        br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])   # 预测右下角目标embedding向量的分支  # conv+bn+relu + 1*1conv

        tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])   # 预测左上角目标偏移的分支（2是要预测xy坐标，因而输出通道数为2）   # conv+bn+relu + 1*1conv
        br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])   # 预测右下角目标偏移的分支   # conv+bn+relu + 1*1conv

        super(model, self).__init__(hgs, tl_modules, br_modules, tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs)  # 调用基类hg_net的__init__函数

        self.loss = CornerNet_Loss(pull_weight=1e-1, push_weight=1e-1)   # 损失函数
```
</details>


### P3.2.2 convolution

实际为conv+bn+relu，位于core/models/py_utils/utils.py

<details>

```python
class convolution(nn.Module):   # conv+bn+relu
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu
```
</details>


### P3.2.3 残差residual

实际为conv+bn + (conv+bn+relu+conv+bn)) + relu，位于core/models/py_utils/utils.py

<details>

```python
class residual(nn.Module):  # 残差连接模块 (conv+bn + (conv+bn+relu+conv+bn)) + relu
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)
```
</details>


### P3.2.4 corner_pool

CornerNet提出了corner池化（corner_pool），位于core/models/py_utils/utils.py

<details>

```python
class corner_pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        self.p1_conv1 = convolution(3, dim, 128)   # conv+bn+relu
        self.p2_conv1 = convolution(3, dim, 128)   # conv+bn+relu

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)   # conv+bn+relu

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool1: TopPool or BottomPool
        # pool2: LeftPool or RightPool
        # 
        #         conv+bn+relu → pool1 
        #      ↗                      ↘ 
        #    x →  conv+bn+relu → pool2 →  + → conv → bn ↘
        #      ↘                                          + → relu → conv+bn+relu
        #         conv → bn  -------------------------→ ↗
        
        p1_conv1 = self.p1_conv1(x)  # pool 1
        pool1    = self.pool1(p1_conv1)

        p2_conv1 = self.p2_conv1(x)  # pool 2
        pool2    = self.pool2(p2_conv1)

        p_conv1 = self.p_conv1(pool1 + pool2)  # pool 1 + pool 2
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2
```
</details>


#### P3.2.4.1 top_pool

实际的top_pool位于core/models/py_utils/_cpools/src/top_pool.cpp。下面为前向计算的代码（反向计算没看）：

<details>

```cpp
std::vector<at::Tensor> top_pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    // Get height
    int64_t height = input.size(2);

    output.copy_(input);
    // 取H方向当前位置值为：从当前位置往下的最大值
    for (int64_t ind = 1; ind < height; ind <<= 1) {  // slice(Tensor, dim, start, end, step = 1)   max_out(out, Tensor, other)
        at::Tensor max_temp = at::slice(output, 2, 0, height-ind);  //  BCHW，dim=2，则在H上，start=0，end=height-ind
        at::Tensor cur_temp = at::slice(output, 2, 0, height-ind);
        at::Tensor next_temp = at::slice(output, 2, ind, height);   //  BCHW，dim=2，则在H上，start=ind，end=height
        at::max_out(max_temp, cur_temp, next_temp);  // 依次取out[i] = max(out[i], out[i+1])，由于每次都比较过相应值，因而上面可以在直接<<
    }

    return { 
        output
    };
}
```
</details>

#### P3.2.4.2 bottom_pool

实际的bottom_pool位于core/models/py_utils/_cpools/src/bottom_pool.cpp。下面为前向计算的代码（反向计算没看）：

<details>

```cpp
std::vector<at::Tensor> pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    // Get height
    int64_t height = input.size(2);

    output.copy_(input);
    // 取H方向当前位置值为：从当前位置往上的最大值
    for (int64_t ind = 1; ind < height; ind <<= 1) {    // slice(Tensor, dim, start, end, step = 1)   max_out(out, Tensor, other)
        at::Tensor max_temp = at::slice(output, 2, ind, height);   //  BCHW，dim=2，则在H上，start=ind，end=height
        at::Tensor cur_temp = at::slice(output, 2, ind, height);
        at::Tensor next_temp = at::slice(output, 2, 0, height-ind);   //  BCHW，dim=2，则在H上，start=0，end=height-ind
        at::max_out(max_temp, cur_temp, next_temp);    // 依次取out[i] = max(out[i], out[i+1])，由于每次都比较过相应值，因而上面可以在直接<<
    }

    return { 
        output
    };
}
```
</details>

#### P3.2.4.3 left_pool

实际的left_pool位于core/models/py_utils/_cpools/src/left_pool.cpp。下面为前向计算的代码（反向计算没看）：

<details>

```cpp
std::vector<at::Tensor> pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    // Get width
    int64_t width = input.size(3);

    output.copy_(input);
    // 取W方向当前位置值为：从当前位置往右的最大值
    for (int64_t ind = 1; ind < width; ind <<= 1) {   // slice(Tensor, dim, start, end, step = 1)   max_out(out, Tensor, other)
        at::Tensor max_temp = at::slice(output, 3, 0, width-ind);     //  BCHW，dim=3，则在W上，start=0，end=width-ind
        at::Tensor cur_temp = at::slice(output, 3, 0, width-ind);        
        at::Tensor next_temp = at::slice(output, 3, ind, width);   //  BCHW，dim=3，则在W上，start=ind，end=height
        at::max_out(max_temp, cur_temp, next_temp);    // 依次取out[i] = max(out[i], out[i+1])，由于每次都比较过相应值，因而上面可以在直接<<
    }

    return { 
        output
    };
}
```
</details>

#### P3.2.4.4 top_pool

实际的top_pool位于core/models/py_utils/_cpools/src/top_pool.cpp。下面为前向计算的代码（反向计算没看）：

<details>

```cpp
std::vector<at::Tensor> pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);

    // Get width
    int64_t width = input.size(3);

    output.copy_(input);
    // 取W方向当前位置值为：从当前位置往左的最大值
    for (int64_t ind = 1; ind < width; ind <<= 1) {    // slice(Tensor, dim, start, end, step = 1)   max_out(out, Tensor, other)
        at::Tensor max_temp = at::slice(output, 3, ind, width);    //  BCHW，dim=3，则在W上，start=ind，end=width
        at::Tensor cur_temp = at::slice(output, 3, ind, width);        
        at::Tensor next_temp = at::slice(output, 3, 0, width-ind);    //  BCHW，dim=3，则在W上，start=0，end=height-ind
        at::max_out(max_temp, cur_temp, next_temp);    // 依次取out[i] = max(out[i], out[i+1])，由于每次都比较过相应值，因而上面可以在直接<<
    }

    return { 
        output
    };
}
```
</details>


### P3.2.5 骨干网络

位于core/models/py_utils/modules.py

<details>

```python
def _make_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def _make_layer_revr(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)

def _make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def _make_unpool_layer(dim):
    return upsample(scale_factor=2)

def _make_merge_layer(dim):
    return merge()

class hg_module(nn.Module):
    def __init__(
        self, n, dims, modules, make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer, make_hg_layer=_make_layer,
        make_low_layer=_make_layer, make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer, make_merge_layer=_make_merge_layer
    ):
        super(hg_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n    = n
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = hg_module(
            n - 1, dims[1:], modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2  = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merg = self.merg(up1, up2)
        return merg

class hg(nn.Module):
    # hg如下
    #                  out[0]                                                 out[0]
    #                   ↑
    #       hgs[0] → cnvs[0] → cnvs_[0]
    #     ↗                            ↘                                      ↑
    # pre                                  + → relu → inters[0] → hgs[1] → cnvs[1]
    #    ↘         inters_[0]          ↗  
    
    def __init__(self, pre, hg_modules, cnvs, inters, cnvs_, inters_):
        super(hg, self).__init__()

        self.pre  = pre
        self.hgs  = hg_modules
        self.cnvs = cnvs

        self.inters  = inters
        self.inters_ = inters_
        self.cnvs_   = cnvs_

    def forward(self, x):
        inter = self.pre(x)

        cnvs  = []
        for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):  # self.hgs为2个hg_module，self.cnvs为2个convolution
            hg  = hg_(inter)
            cnv = cnv_(hg)
            cnvs.append(cnv)

            if ind < len(self.hgs) - 1:    # len(self.hgs)-1=1，此处执行一次
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = nn.functional.relu_(inter)
                inter = self.inters[ind](inter)
        return cnvs

class hg_net(nn.Module):
    def __init__(self, hg, tl_modules, br_modules, tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs):
        super(hg_net, self).__init__()

        self._decode = _decode

        self.hg = hg

        self.tl_modules = tl_modules
        self.br_modules = br_modules

        self.tl_heats = tl_heats
        self.br_heats = br_heats

        self.tl_tags = tl_tags
        self.br_tags = br_tags
        
        self.tl_offs = tl_offs
        self.br_offs = br_offs

    def _train(self, *xs):
        # train时由于hg有2个输出，因而右侧每组都有2个结果，test时只使用最后一个hg的输出进行预测
        #                               ↗tl_heat   左上角热图
        #                → tl_modules → → tl_tags   左上角embedding向量
        #              ↗               ↘tl_offs   左上角偏移
        # image → hg → 
        #              ↘               ↗br_heat   右下角热图
        #                → br_modules → → br_tags   右下角embedding向量 
        #                               ↘br_offs   右下角偏移
        image = xs[0]    # 
        cnvs  = self.hg(image)  # 通过hg，得到特征

        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]   # 左上角池化
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]   # 右下角池化
        tl_heats   = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]   # 左上角热图
        br_heats   = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]   # 右下角热图
        tl_tags    = [tl_tag_(tl_mod)  for tl_tag_,  tl_mod in zip(self.tl_tags,  tl_modules)]   # 左上角embedding向量
        br_tags    = [br_tag_(br_mod)  for br_tag_,  br_mod in zip(self.br_tags,  br_modules)]   # 右下角embedding向量
        tl_offs    = [tl_off_(tl_mod)  for tl_off_,  tl_mod in zip(self.tl_offs,  tl_modules)]   # 左上角偏移
        br_offs    = [br_off_(br_mod)  for br_off_,  br_mod in zip(self.br_offs,  br_modules)]   # 右下角偏移
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs]   # 左上角热图、右下角热图、左上角embedding向量、右下角embedding向量、左上角偏移、右下角偏移

    def _test(self, *xs, **kwargs):   # test时只使用最后一个hg的输出进行预测
        image = xs[0]
        cnvs  = self.hg(image)    # 通过hg，得到特征

        tl_mod = self.tl_modules[-1](cnvs[-1])   # 左上角池化
        br_mod = self.br_modules[-1](cnvs[-1])   # 右下角池化

        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)   # 左上角、右下角热图
        tl_tag,  br_tag  = self.tl_tags[-1](tl_mod),  self.br_tags[-1](br_mod)    # 左上角embedding向量、右下角embedding向量
        tl_off,  br_off  = self.tl_offs[-1](tl_mod),  self.br_offs[-1](br_mod)    # 左上角、右下角偏移

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]

        #  self._decode得到检测结果  [B*num_dets*8]  8为：bbox坐标，目标中心得分，左上角目标分值，右下角目标分值，目标类别
        return self._decode(*outs, **kwargs), tl_heat, br_heat, tl_tag, br_tag  #  检测结果，左上角热图、右下角热图、左上角embedding向量、右下角embedding向量  最终返回core\test\cornernet.py中decode函数（只使用本函数结果的[0]，中间还有一系列返回位置）

    def forward(self, *xs, test=False, **kwargs):
        if not test:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)
```
</details>


### P3.2.6 额外调用的函数upsample和merge

<details>

```python
class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

class merge(nn.Module):
    def forward(self, x, y):
        return x + y
```
</details>


## P3.3 损失CornerNet_Loss

位于core/models/py_utils/losses.py

### P3.3.1 CornerNet_Loss

<details>

```python
def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()  # mask：B*128  num：当前batch每个图像上有多少个目标
    tag0 = tag0.squeeze()   # B*128*1  tag0：去掉最后为1的维度  为上（或左）的embedding向量
    tag1 = tag1.squeeze()   # B*128*1  tag1：去掉最后为1的维度  为下（或右）的embedding向量 

    tag_mean = (tag0 + tag1) / 2   # 中点

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)  # 论文中公式4
    tag0 = tag0[mask].sum()     # 取有效的结果
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)  # 论文中公式4
    tag1 = tag1[mask].sum()     # 取有效的结果
    pull = tag0 + tag1   # 论文中公式4

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)  # B*1*128 + B*128*1的进行broadcast相加，得到B*128*128。
    mask = mask.eq(2)   # 得到两两配对，均是目标的索引
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)  # 得到两两配对时，相互之间的距离
    dist = 1 - torch.abs(dist)  # 论文中公式5的△-|e_k-e_j|，此处包含j=k
    dist = nn.functional.relu(dist, inplace=True)   # 论文中公式5的max(0, △-|e_k-e_j|)，此处包含j=k
    dist = dist - 1 / (num + 1e-4)   # 去除j=k的，是需要除法，此处每个值都-1，下面sum()后，相当于(dist-1/num)*(num*num)=系数*dist-num，实际每个图像共num个j=k的，此处结合dist.sum()等效于dist-num
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()   # 论文中公式5
    return pull, push

def _off_loss(off, gt_off, mask):  # gt_off：B*128*2   mask:B*128，有目标相应值为1，否则为0
    num  = mask.float().sum()  # 
    mask = mask.unsqueeze(2).expand_as(gt_off)   # B*128*2 

    off    = off[mask]      # 得到有目标的预测偏移
    gt_off = gt_off[mask]   # 得到有目标的实际偏移
    
    off_loss = nn.functional.smooth_l1_loss(off, gt_off, reduction="sum")   # 预测偏移和实际偏移的L1损失
    off_loss = off_loss / (num + 1e-4)   # L1损失
    return off_loss

def _focal_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:   # preds为list，代表预测的[上，左]或者[下，右]热图
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class CornerNet_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss):
        super(CornerNet_Loss, self).__init__()

        self.pull_weight = pull_weight   # 分组相同角点的损失的权重
        self.push_weight = push_weight   # 分离不同角点的损失的权重  
        self.off_weight  = off_weight    # 预测偏移损失的权重
        self.focal_loss  = focal_loss    # 热图的损失函数
        self.ae_loss     = _ae_loss      # 对角点进行分组的损失函数
        self.off_loss    = _off_loss     # 预测偏移的损失函数

    def forward(self, outs, targets):
        tl_heats = outs[0]   # 预测目标左上角热图   2个B*80*H*W
        br_heats = outs[1]   # 预测目标右下角热图   2个B*80*H*W
        tl_tags  = outs[2]   # 预测目标左上角embedding向量   2个B*1*H*W
        br_tags  = outs[3]   # 预测目标右下角embedding向量   2个B*1*H*W
        tl_offs  = outs[4]   # 预测目标左上角偏移   2个B*2*H*W
        br_offs  = outs[5]   # 预测目标右下角偏移   2个B*2*H*W

        gt_tl_heat  = targets[0]     # BCHW      gt左上角热图  C=80
        gt_br_heat  = targets[1]     # BCHW      gt右下角热图  C=80
        gt_mask     = targets[2]     # B*128     每个图像上有目标的mask，有目标相应值为1，否则为0
        gt_tl_off   = targets[3]     # B*128*2   左上角坐标gt偏移
        gt_br_off   = targets[4]     # B*128*2   右下角坐标gt偏移
        gt_tl_ind   = targets[5]     # B*128     当前图像上，当宽高为一维时，左上角顶点的位置
        gt_br_ind   = targets[6]     # B*128     当前图像上，当宽高为一维时，右下角顶点的位置

        # focal loss
        focal_loss = 0  # 预测角点的损失（左上角+右下角）

        tl_heats = [_sigmoid(t) for t in tl_heats]   # 左上角热图映射到0-1之间
        br_heats = [_sigmoid(b) for b in br_heats]   # 右下角热图映射到0-1之间

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)   # 左上角热图的总损失
        focal_loss += self.focal_loss(br_heats, gt_br_heat)   # 之前损失 + 右下角热图的总损失

        # tag loss
        pull_loss = 0
        push_loss = 0
        tl_tags   = [_tranpose_and_gather_feat(tl_tag, gt_tl_ind) for tl_tag in tl_tags]  # 分别从左上角embedding向量上得到实际点的embedding向量，2个B*128*1的矩阵
        br_tags   = [_tranpose_and_gather_feat(br_tag, gt_br_ind) for br_tag in br_tags]  # 分别从右下角embedding向量上得到实际点的embedding向量，2个B*128*1的矩阵
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)  # 依次计算是否是同一个目标框的上和下、左和右角点的损失
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss   # 分组相同角点的损失
        push_loss = self.push_weight * push_loss   # 分离不同角点的损失

        off_loss = 0
        tl_offs  = [_tranpose_and_gather_feat(tl_off, gt_tl_ind) for tl_off in tl_offs]  # 预测左上角角点的实际偏移，2个B*128*2的矩阵
        br_offs  = [_tranpose_and_gather_feat(br_off, gt_br_ind) for br_off in br_offs]  # 预测右下角角点的实际偏移，2个B*128*2的矩阵
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_mask)     # 预测左上角角点偏移的损失
            off_loss += self.off_loss(br_off, gt_br_off, gt_mask)     # 预测右下角角点偏移的损失
        off_loss = self.off_weight * off_loss    # 总的预测偏移的损失

        loss = (focal_loss + pull_loss + push_loss + off_loss) / max(len(tl_heats), 1)    # 总损失
        return loss.unsqueeze(0)
```
</details>


### P3.3.2 _tranpose_and_gather_feat

位于core/models/py_utils/utils.py

<details>

```python
def _tranpose_and_gather_feat(feat, ind):   # 和_gather_feat比，有一个BCHW转到BHWC再转到B(HW)C的过程
    feat = feat.permute(0, 2, 3, 1).contiguous()   # BCHW变成BHWC
    feat = feat.view(feat.size(0), -1, feat.size(3))    # BHWC变成B(HW)C
    feat = _gather_feat(feat, ind)   # 得到当前batch中HW上所有ind处的特征   B*n*C
    return feat

def _gather_feat(feat, ind, mask=None):    # 和_tranpose_and_gather_feat比，无一个BCHW转到BHWC再转到B(HW)C的过程
    dim  = feat.size(2)  # 通道数
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)   # B*n*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
```
</details>


## P3.4 获取数据时的cornernet

位于core/sample/cornernet.py

### P3.4.1 cornernet

<details>

```python
def _resize_image(image, detections, size):  # 缩放图像及gt信息
    detections    = detections.copy()
    height, width = image.shape[0:2]   # 输入图像高宽
    new_height, new_width = size   # 目标高宽

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height   # 高度缩放比例
    width_ratio  = new_width  / width    # 宽度缩放比例
    detections[:, 0:4:2] *= width_ratio   # 缩放相应gt信息
    detections[:, 1:4:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):   # 裁剪（宽高在图像范围内）、校验（目标宽高均>0）gt信息
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)   # 保证gt信息不超过图像边界，防止之前随机裁剪把目标裁掉了
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & ((detections[:, 3] - detections[:, 1]) > 0)   # 目标宽高均>0的索引，防止之前随机裁剪把目标裁掉了
    detections = detections[keep_inds]   # 得到处理后的gt信息
    return detections

def cornernet(system_configs, db, k_ind, data_aug, debug):   # cornernet获取数据的函数
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]  # 80
    input_size   = db.configs["input_size"]  # [511, 511]
    output_size  = db.configs["output_sizes"][0]   # output_sizes：[[128, 128]]，取[0]得到[128, 128]

    border        = db.configs["border"]   # 128
    lighting      = db.configs["lighting"] # True   core\dbs\detection.py
    rand_crop     = db.configs["rand_crop"]  # true
    rand_color    = db.configs["rand_color"]  # true
    rand_scales   = db.configs["rand_scales"]  # 配置为null，则为np.arange(rand_scale_min=0.6, rand_scale_max=1.4, rand_scale_step=0.1)
    gaussian_bump = db.configs["gaussian_bump"]   # true
    gaussian_iou  = db.configs["gaussian_iou"]   # 0.3
    gaussian_rad  = db.configs["gaussian_radius"]  # -1   core\dbs\detection.py

    max_tag_len = 128

    # allocating memory
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)   # BCHW  图像
    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)   # BCHW  左上角顶点热图
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)   # BCHW  右下角顶点热图
    tl_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)                               # B*128*2   左上角坐标gt偏移
    br_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)                               # B*128*2   右下角坐标gt偏移
    tl_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)                                    # B*128     当前图像上，当宽高为一维时，左上角顶点的位置
    br_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)                                    # B*128     当前图像上，当宽高为一维时，右下角顶点的位置
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8)                                    # B*128     每个图像上有目标的mask，有目标相应值为1，否则为0
    tag_lens    = np.zeros((batch_size, ), dtype=np.int32)                                               # 128       当前batch中每个图像上目标数量

    db_size = db.db_inds.size   # 数据库中图像数量
    for b_ind in range(batch_size):  # 得到batch中的数据
        if not debug and k_ind == 0:
            db.shuffle_inds()    # 打乱图像索引顺序，得到新的图像顺序

        db_ind = db.db_inds[k_ind]    # 得到当前图像索引
        k_ind  = (k_ind + 1) % db_size  # 超出图像数量，则从0开始

        image_path = db.image_path(db_ind)   # 图像路径
        image      = cv2.imread(image_path)  # reading image

        detections = db.detections(db_ind)    # detections:{图像名字:n*[x1,y1,x2,y2,类别]}  # reading detections  

        if not debug and rand_crop:   # cropping an image randomly
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)  # 随机裁剪图像，返回裁剪后的图像及gt信息（n*[x1,y1,x2,y2,类别]）

        image, detections = _resize_image(image, detections, input_size)    # 缩放图像及gt信息
        detections = _clip_detections(image, detections)     # 裁剪（宽高在图像范围内）、校验（目标宽高均>0）gt信息，防止之前随机裁剪把目标裁掉了

        width_ratio  = output_size[1] / input_size[1]   # 输出特征和输入图像宽高之比
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:   # 水平翻转图像及坐标
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        if not debug:
            image = image.astype(np.float32) / 255.   # 图像像素范围缩放到0-1
            if rand_color:
                color_jittering_(data_rng, image)   # 亮度、对比度、饱和度随机扰动
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)   # 不知道这个是干什么的
            normalize_(image, db.mean, db.std)    # 归一化图像
        images[b_ind] = image.transpose((2, 0, 1))   # CHW到HWC

        for ind, detection in enumerate(detections):  # 依次处理每个目标，得到训练用的gt相关结果
            category = int(detection[-1]) - 1    # 目标类别。coco的0为背景，cornernet无背景通道

            xtl, ytl = detection[0], detection[1]  # 左上角顶点xy坐标
            xbr, ybr = detection[2], detection[3]  # 右下角顶点xy坐标

            fxtl = (xtl * width_ratio)  # 输出特征左上角xy坐标
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)  # 输出特征右下角xy坐标
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)   # 左上角顶点xy坐标取整
            ytl = int(fytl)
            xbr = int(fxbr)   # 右下角顶点xy坐标取整
            ybr = int(fybr)

            if gaussian_bump:   # True
                width  = detection[2] - detection[0]   # 目标宽高
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)   # 特征图上目标宽高
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)   # 半径。在半径内的点得到的框仍旧认为是正确的检测框
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad

                draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)  # 热图上相应通道相应顶点画半径为radius的高斯分布的gt热图
                draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
            else:
                tl_heatmaps[b_ind, category, ytl, xtl] = 1   # 热图上相应通道相应顶点设置为1
                br_heatmaps[b_ind, category, ybr, xbr] = 1

            tag_ind = tag_lens[b_ind]   # 得到当前图像上第tag_ind个目标的索引
            tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]   # 设置当前图像上相应目标的gt的偏移
            br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
            tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl   # 得到当前图像上，当宽高为一维时，顶点的位置
            br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
            tag_lens[b_ind] += 1   # 相应图像上目标数量+1

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]        #  得到每个图像上目标数量
        tag_masks[b_ind, :tag_len] = 1   # 每个图像上有目标的mask，有目标相应值为1，否则为0

    images      = torch.from_numpy(images)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    tl_regrs    = torch.from_numpy(tl_regrs)
    br_regrs    = torch.from_numpy(br_regrs)
    tl_tags     = torch.from_numpy(tl_tags)
    br_tags     = torch.from_numpy(br_tags)
    tag_masks   = torch.from_numpy(tag_masks)

    return {"xs": [images], "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags]}, k_ind
```
</details>


### P3.4.2 random_crop

随机裁剪图像，位于core/sample/utils.py

<details>

```python
def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

def random_crop(image, detections, random_scales, view_size, border=64):  # 随机裁剪图像
    view_height, view_width   = view_size   # 裁剪后图像的高宽
    image_height, image_width = image.shape[0:2]

    scale  = np.random.choice(random_scales)   # 在random_scales随机选一个scale
    height = int(view_height * scale)   # 裁剪并缩放后的高宽
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)   # 裁剪后的图像

    w_border = _get_border(border, image_width)   # 不懂为何这样算？？？保证裁剪前图像至少宽度是2*w_border（不一定用到这么多像素，只是保证有）
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)   # 裁剪前图像中心的位置
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)  # 裁剪前的起点和终点
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx   # 裁剪前图像上实际裁剪的图像宽高
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2   # 裁剪后图像的中心
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)  # 返回指定如何对序列进行裁切的对象  slice(start, end, step=1)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]   # 裁剪图像

    # crop detections
    cropped_detections = detections.copy()   # 复制gt结果  n*[x1,y1,x2,y2,类别]
    cropped_detections[:, 0:4:2] -= x0   # 减去原始图像上的偏移
    cropped_detections[:, 1:4:2] -= y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w  # 加上在目标图像上的偏移
    cropped_detections[:, 1:4:2] += cropped_cty - top_h

    return cropped_image, cropped_detections  # 返回裁剪后的图像和gt信息（n*[x1,y1,x2,y2,类别]）
```
</details>


### P3.4.3 draw_gaussian

用于在热图相应位置画高斯分布的gt热图。位于core/sample/utils.py

<details>

```python
def draw_gaussian(heatmap, center, radius, k=1):  # 热图上相应通道heatmap相应顶点center画半径为radius的高斯分布的gt热图
    diameter = 2 * radius + 1  # 直径
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)    # 得到中心在原点，直径为diameter的高斯分布

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]   # 取热图相应位置，实际为引用
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]    # 取高斯分布相应位置，实际为引用   radius为高斯分布的中心
np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)   # 取热图相应位置和高斯分布相应位置的最大值，输出到masked_heatmap，由于上面为引用，因而修改的是heatmap相应位置

def gaussian2D(shape, sigma=1):  # 得到中心在原点，直径为shape的高斯分布
    m, n = [(ss - 1.) / 2. for ss in shape]   # 半径
    y, x = np.ogrid[-m:m+1,-n:n+1]   # 坐标

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))   # 高斯分布的矩阵
    h[h < np.finfo(h.dtype).eps * h.max()] = 0   # 过小的值设置为0
    return h
```
</details>

### P3.4.4 gaussian_radius

位于core/sample/utils.py

<details>

```python
def gaussian_radius(det_size, min_overlap):
    height, width = det_size   # 特征图上目标的高宽。不知道下面为何这样算？？？

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)
```
</details>


### P3.4.5 gaussian_radius

归一化图像。位于core/sample/utils.py

<details>

```python
def normalize_(image, mean, std):   # 归一化图像
    image -= mean
    image /= std
```
</details>


### P3.4.6 color_jittering_

对图像亮度、对比度、饱和度进行扰动。位于core/sample/utils.py

<details>

```python
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):   # 饱和度随机扰动
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):  # 亮度随机扰动
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):  # 对比度随机扰动
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_jittering_(data_rng, image):   # 亮度、对比度、饱和度随机扰动
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
```
</details>


### P3.4.7 lighting_

位于core/sample/utils.py

<details>

```python
def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)
```
</details>


## P3.5 获取COCO数据

位于core/dbs/coco.py

### P3.5.1 COCO

<details>

```python
class COCO(DETECTION):
    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(COCO, self).__init__(db_config)   # 调用基类DETECTION的__init__函数，更新其self._configs相关参数

        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352], [-0.5832747, 0.00994535, -0.81221408], [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self._coco_cls_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90
        ]

        self._coco_cls_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse','sheep', 'cow', 'elephant', 
            'bear', 'zebra','giraffe', 'backpack', 'umbrella', 
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
            'snowboard','sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self._cls2coco  = {ind + 1: coco_id for ind, coco_id in enumerate(self._coco_cls_ids)}
        self._coco2cls  = {coco_id: cls_id for cls_id, coco_id in self._cls2coco.items()}
        self._coco2name = {cls_id: cls_name for cls_id, cls_name in zip(self._coco_cls_ids, self._coco_cls_names)}
        self._name2coco = {cls_name: cls_id for cls_name, cls_id in self._coco2name.items()}

        if split is not None:
            coco_dir = os.path.join(sys_config.data_dir, "coco")

            self._split     = {"trainval": "trainval2014", "minival":  "minival2014", "testdev":  "testdev2017"}[split]
            self._data_dir  = os.path.join(coco_dir, "images", self._split)
            self._anno_file = os.path.join(coco_dir, "annotations", "instances_{}.json".format(self._split))

            self._detections, self._eval_ids = self._load_coco_annos()  # detections:{图像名字:n*[x1,y1,x2,y2,类别]}
            self._image_ids = list(self._detections.keys())
            self._db_inds   = np.arange(len(self._image_ids))   # 图像索引

    def _load_coco_annos(self):
        from pycocotools.coco import COCO

        coco = COCO(self._anno_file)
        self._coco = coco

        class_ids = coco.getCatIds()
        image_ids = coco.getImgIds()
        
        eval_ids   = {}
        detections = {}
        for image_id in image_ids:
            image = coco.loadImgs(image_id)[0]
            dets  = []
            
            eval_ids[image["file_name"]] = image_id
            for class_id in class_ids:
                annotation_ids = coco.getAnnIds(imgIds=image["id"], catIds=class_id)
                annotations    = coco.loadAnns(annotation_ids)
                category       = self._coco2cls[class_id]
                for annotation in annotations:
                    det     = annotation["bbox"] + [category]
                    det[2] += det[0]
                    det[3] += det[1]
                    dets.append(det)   # [x1,y1,x2,y2,类别]

            file_name = image["file_name"]
            if len(dets) == 0:
                detections[file_name] = np.zeros((0, 5), dtype=np.float32)
            else:
                detections[file_name] = np.array(dets, dtype=np.float32)
        return detections, eval_ids  # detections:{图像名字:n*[x1,y1,x2,y2,类别]}

    def image_path(self, ind):
        if self._data_dir is None:
            raise ValueError("Data directory is not set")

        db_ind    = self._db_inds[ind]    # 得到当前图像索引
        file_name = self._image_ids[db_ind]   # 得到图像名字
        return os.path.join(self._data_dir, file_name)

    def detections(self, ind):
        db_ind    = self._db_inds[ind]     # 得到当前图像索引
        file_name = self._image_ids[db_ind]   # 得到图像名字
        return self._detections[file_name].copy()   # 得到图像gt信息  detections:{图像名字:n*[x1,y1,x2,y2,类别]}

    def cls2name(self, cls):
        coco = self._cls2coco[cls]
        return self._coco2name[coco]

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._cls2coco[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox  = list(map(self._to_float, bbox[0:4]))

                    detection = {"image_id": coco_id, "category_id": category_id, "bbox": bbox, "score": float("{:.2f}".format(score))}

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids):
        from pycocotools.cocoeval import COCOeval

        if self._split == "testdev":
            return None

        coco = self._coco

        eval_ids = [self._eval_ids[image_id] for image_id in image_ids]
        cat_ids  = [self._cls2coco[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]
```
</details>


### P3.5.2 DETECTION

COCO继承自DETECTION，其位于core/dbs/detection.py。包括如下参数：

<details>

```python
class DETECTION(BASE):
    def __init__(self, db_config):
        super(DETECTION, self).__init__()

        # Configs for training
        self._configs["categories"]      = 80
        self._configs["rand_scales"]     = [1]
        self._configs["rand_scale_min"]  = 0.8
        self._configs["rand_scale_max"]  = 1.4
        self._configs["rand_scale_step"] = 0.2

        # Configs for both training and testing
        self._configs["input_size"]      = [383, 383]
        self._configs["output_sizes"]    = [[96, 96], [48, 48], [24, 24], [12, 12]]

        self._configs["score_threshold"] = 0.05
        self._configs["nms_threshold"]   = 0.7
        self._configs["max_per_set"]     = 40
        self._configs["max_per_image"]   = 100
        self._configs["top_k"]           = 20
        self._configs["ae_threshold"]    = 1
        self._configs["nms_kernel"]      = 3
        self._configs["num_dets"]        = 1000

        self._configs["nms_algorithm"]   = "exp_soft_nms"
        self._configs["weight_exp"]      = 8
        self._configs["merge_bbox"]      = False

        self._configs["data_aug"]        = True
        self._configs["lighting"]        = True

        self._configs["border"]          = 64
        self._configs["gaussian_bump"]   = False
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1
        self._configs["rand_crop"]       = False
        self._configs["rand_color"]      = False
        self._configs["rand_center"]     = True

        self._configs["init_sizes"]      = [192, 255]
        self._configs["view_sizes"]      = []

        self._configs["min_scale"]       = 16
        self._configs["max_scale"]       = 32

        self._configs["att_sizes"]       = [[16, 16], [32, 32], [64, 64]]
        self._configs["att_ranges"]      = [[96, 256], [32, 96], [0, 32]]
        self._configs["att_ratios"]      = [16, 8, 4]
        self._configs["att_scales"]      = [1, 1.5, 2]
        self._configs["att_thresholds"]  = [0.3, 0.3, 0.3, 0.3]
        self._configs["att_nms_ks"]      = [3, 3, 3]
        self._configs["att_max_crops"]   = 8
        self._configs["ref_dets"]        = True

        # Configs for testing
        self._configs["test_scales"]     = [1]
        self._configs["test_flipped"]    = True

        self.update_config(db_config)   # 根据输入的config，更新self._configs相关参数

        if self._configs["rand_scales"] is None:
            self._configs["rand_scales"] = np.arange(self._configs["rand_scale_min"], self._configs["rand_scale_max"], self._configs["rand_scale_step"])  # 随机取一个值
```
</details>


### P3.5.3 BASE

DETECTION继承自BASE，其位于core/dbs/base.py，如下：

<details>

```python
class BASE(object):
    def __init__(self):
        self._split     = None
        self._db_inds   = []
        self._image_ids = []

        self._mean    = np.zeros((3, ), dtype=np.float32)
        self._std     = np.ones((3, ), dtype=np.float32)
        self._eig_val = np.ones((3, ), dtype=np.float32)
        self._eig_vec = np.zeros((3, 3), dtype=np.float32)

        self._configs = {}
        self._configs["data_aug"] = True

        self._data_rng = None

    @property
    def configs(self):
        return self._configs

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def db_inds(self):
        return self._db_inds

    @property
    def split(self):
        return self._split

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

    def image_ids(self, ind):
        return self._image_ids[ind]

    def image_path(self, ind):
        pass

    def write_result(self, ind, all_bboxes, all_scores):
        pass

    def evaluate(self, name):
        pass

    def shuffle_inds(self, quiet=False):   # 打乱图像索引顺序，得到新的图像顺序
        if self._data_rng is None:
            self._data_rng = np.random.RandomState(os.getpid())

        if not quiet:
            print("shuffling indices...")
        rand_perm = self._data_rng.permutation(len(self._db_inds))   # _db_inds应该是数据库中图像索引：np.arange(len(self._image_ids))，打乱_db_inds顺序，起到随机的作用
        self._db_inds = self._db_inds[rand_perm]   # 得到新的图像索引
```
</details>


## P3.6 evaluate时的cornernet

hg_net在test（包括evaluate）时，调用_test函数，该函数调用_decode，最终返回core\test\cornernet.py中decode函数。

### P3.6.1 cornernet

<details>

```python
def rescale_dets_(detections, ratios, borders, sizes):  # 缩放检测结果
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def decode(nnet, images, K, ae_threshold=0.5, kernel=3, num_dets=1000):
    # 检测，最终调用core\models\py_utils\modules.py中_test
    # 得到：检测结果，左上角热图、右下角热图、左上角embedding向量、右下角embedding向量 
    # 其中，检测结果  [B*num_dets*8]  8为：bbox坐标，目标中心得分，左上角目标分值，右下角目标分值，目标类别
    detections = nnet.test([images], ae_threshold=ae_threshold, test=True, K=K, kernel=kernel, num_dets=num_dets)[0]  # 得到[B*num_dets*8]的检测结果 8为：bbox坐标，目标中心得分，左上角目标分值，右下角目标分值，目标类别
    return detections.data.cpu().numpy()

def cornernet(db, nnet, result_dir, debug=False, decode_func=decode):   # test时会调用本函数
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval2014":
        db_inds = db.db_inds[:100] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]

    num_images = db_inds.size   # 图像数量
    categories = db.configs["categories"]

    timer = Timer()
    top_bboxes = {}
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[ind]

        image_id   = db.image_ids(db_ind)
        image_path = db.image_path(db_ind)
        image      = cv2.imread(image_path)   # 读取图像

        timer.tic()
        top_bboxes[image_id] = cornernet_inference(db, nnet, image)  # 前向推断函数，得到bbox+得分
        timer.toc()

        if debug:
            image_path = db.image_path(db_ind)
            image      = cv2.imread(image_path)
            bboxes     = {db.cls2name(j): top_bboxes[image_id][j] for j in range(1, categories + 1)}
            image      = draw_bboxes(image, bboxes)
            debug_file = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            cv2.imwrite(debug_file, image)
    print('average time: {}'.format(timer.average_time))

    result_json = os.path.join(result_dir, "results.json")
    detections  = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)

    cls_ids   = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    db.evaluate(result_json, cls_ids, image_ids)
    return 0

def cornernet_inference(db, nnet, image, decode_func=decode):
    K             = db.configs["top_k"]
    ae_threshold  = db.configs["ae_threshold"]
    nms_kernel    = db.configs["nms_kernel"]
    num_dets      = db.configs["num_dets"]
    test_flipped  = db.configs["test_flipped"]

    input_size    = db.configs["input_size"]
    output_size   = db.configs["output_sizes"][0]
    
    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {"nms": 0, "linear_soft_nms": 1, "exp_soft_nms": 2}[db.configs["nms_algorithm"]]

    height, width = image.shape[0:2]

    height_scale  = (input_size[0] + 1) // output_size[0]  # 特征图相比他图像，宽高缩放倍数
    width_scale   = (input_size[1] + 1) // output_size[1]

    im_mean = torch.cuda.FloatTensor(db.mean).reshape(1, 3, 1, 1)
    im_std  = torch.cuda.FloatTensor(db.std).reshape(1, 3, 1, 1)

    detections = []
    for scale in scales:   # 多尺度检测，增加性能
        new_height = int(height * scale)
        new_width  = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width  = new_width  | 127

        images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios  = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes   = np.zeros((1, 2), dtype=np.float32)

        out_height, out_width = (inp_height + 1) // height_scale, (inp_width + 1) // width_scale
        height_ratio = out_height / inp_height
        width_ratio  = out_width  / inp_width

        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])  # 裁剪图像

        resized_image = resized_image / 255.   # 缩放图像像素到0-1

        images[0]  = resized_image.transpose((2, 0, 1))   # HWC到CHW
        borders[0] = border
        sizes[0]   = [int(height * scale), int(width * scale)]
        ratios[0]  = [height_ratio, width_ratio]

        if test_flipped:   # 水平镜像图像
            images  = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images  = torch.from_numpy(images).cuda()
        images -= im_mean   # 归一化图像
        images /= im_std
        
        dets = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel, num_dets=num_dets)  # 得到[B*num_dets*8]的检测结果 8为：bbox坐标，目标中心得分，左上角目标分值，右下角目标分值，目标类别
        if test_flipped:  # 水平翻转x坐标
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets = dets.reshape(1, -1, 8)

        rescale_dets_(dets, ratios, borders, sizes)    # 缩放检测结果
        dets[:, :, 0:4] /= scale
        detections.append(dets)

    detections = np.concatenate(detections, axis=1)  # 将多尺度的检测结果拼接  1*(2*K)*8

    classes    = detections[..., -1]  # 检测到的目标类别   1*(2*K)
    classes    = classes[0]           # bs=1，因而得到(2*K)的类别
    detections = detections[0]        # bs=1，因而得到(2*K)的类别   (2*K)*8

    keep_inds  = (detections[:, 4] > -1)  # 目标中心得分大于阈值的索引   # reject detections with negative scores
    detections = detections[keep_inds]    # 过滤后的目标  n*8
    classes    = classes[keep_inds]       # 过滤后的类别  n

    top_bboxes = {}
    for j in range(categories):
        keep_inds = (classes == j)   # 第i类别的索引
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)   # 检测结果
        if merge_bbox:   # 默认false，CornerNet-multi_scale.json时为True
            soft_nms_merge(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
        else:
            soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm)
        top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]   # 最终检测结果   bbox+得分

    scores = np.hstack([top_bboxes[j][:, -1] for j in range(1, categories + 1)])
    if len(scores) > max_per_image:   # 当前图像上监测的目标个数太多，则按目标得分，去除得分太低的目标
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds     = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]
    return top_bboxes
```
</details>


### P3.6.2 得到检测结果_decode

位于core/models/py_utils/utils.py

<details>

```python
def _decode(tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, K=100, kernel=1, ae_threshold=1, num_dets=1000, no_border=False):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)   # 特征归一化
    br_heat = torch.sigmoid(br_heat)

    tl_heat = _nms(tl_heat, kernel=kernel)   # 得到极大值点（其他位置均为0）  perform nms on heatmaps  kernel=3
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)  # 前k个分值最大的目标的分值、在宽高上的索引、类别、y坐标、x坐标，均为b*K
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)   # b*K*K
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if no_border:  # no_border=False
        tl_ys_binds = (tl_ys == 0)
        tl_xs_binds = (tl_xs == 0)
        br_ys_binds = (br_ys == height - 1)
        br_xs_binds = (br_xs == width  - 1)

    if tl_regr is not None and br_regr is not None:  # 预测坐标偏移
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)   # 得到预测目标的坐标偏移   B*K*2
        tl_regr = tl_regr.view(batch, K, 1, 2)   # B*K*1*2
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]   # 预测的坐标  tl_regr[..., 0]取0维度，为B*K*1，和b*K*K的broadast相加，得到b*K*K
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)  # 预测的bbox b*K*K*4   # all possible boxes based on top k corners (ignoring class)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)   # 预测的embedding向量  输入tl_tag为B*1*W*H，输出tl_tag为B*K*1
    tl_tag = tl_tag.view(batch, K, 1)   # B*K*1
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)   # B*1*K
    dists  = torch.abs(tl_tag - br_tag)   # broadcast减法，得到B*K*K，代表左上和右下两两对比的embedding向量之间的距离

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)   # B*K转到B*K*K
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2   # B*K*K，代表左上和右下两两对比中点的得分

    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)  # B*K转到B*K*1扩充到B*K*K  左上角目标类别
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)  # B*K转到B*1*K扩充到B*K*K  右下角目标类别
    cls_inds = (tl_clses != br_clses)   # 代表左上和右下两两对比目标不同的mask，下面用于去除相应的结果    # reject boxes based on classes

    dist_inds = (dists > ae_threshold)    # 左上和右下两两对比的embedding向量之间的距离大于阈值的mask，下面用于去除相应的结果   # reject boxes based on distances

    width_inds  = (br_xs < tl_xs)    # 预测目标右下角小于左上角的mask，下面用于去除相应的结果   # reject boxes based on widths and heights
    height_inds = (br_ys < tl_ys)

    if no_border:
        scores[tl_ys_binds] = -1
        scores[tl_xs_binds] = -1
        scores[br_ys_binds] = -1
        scores[br_xs_binds] = -1

    scores[cls_inds]    = -1   # 去除相应的左上和右下两两对比中点的得分
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)   # B*K*K转到B*(K*K)
    scores, inds = torch.topk(scores, num_dets)  # 最高的num_dets个目标中心得分及相应索引，均为B*num_dets
    scores = scores.unsqueeze(2)   # B*num_dets*1

    bboxes = bboxes.view(batch, -1, 4)  # B*K*K*4转到B*(K*K)*4
    bboxes = _gather_feat(bboxes, inds)  # 取最高的num_dets个目标的bbox  B*num_dets*4

    clses  = tl_clses.contiguous().view(batch, -1, 1)   # 预测的左上角的类别  B*K*K转到B*(K*K)*1
    clses  = _gather_feat(clses, inds).float()  # 取最高的num_dets个类别   B*num_dets*1

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)      # 预测的左上角的前K个目标的分值  B*K*K转到B*(K*K)*1
    tl_scores = _gather_feat(tl_scores, inds).float()          # 取最高的num_dets个目标的分值   B*num_dets*1
    br_scores = br_scores.contiguous().view(batch, -1, 1)      # 预测的右下角的前K个目标的分值  B*K*K转到B*(K*K)*1
    br_scores = _gather_feat(br_scores, inds).float()          # 取最高的num_dets个目标的分值   B*num_dets*1

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)   # 得到检测结果  B*num_dets*8  8为：bbox坐标，目标中心得分，左上角目标分值，右下角目标分值，目标类别
    return detections
```
</details>


### P3.6.3 得到极大值点的_nms

_nms通过max pool得到极大值点，并未进行非极大值抑制。位于core/models/py_utils/utils.py

<details>

```python
def _nms(heat, kernel=1):  # 实际kernel=3
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)  # maxpool，得到每个位置3*3内的最大值，当做极大值点
    keep = (hmax == heat).float()   # 得到极值点mask
    return heat * keep  # 返回极大值点
```
</details>


### P3.6.4 得到前K个得分最高的极值点信息_topk

_topk用于得到前k个分值最大的目标的分值、在宽高上的索引、类别、y坐标、x坐标。位于core/models/py_utils/utils.py

<details>

```python
def _topk(scores, K=20):  # scores BCHW
    batch, cat, height, width = scores.size() 

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)    # 用于得到极值点（即目标）的得分及类别索引  b*K

    topk_clses = (topk_inds / (height * width)).int()   #  极值点（即目标）的类别  b*K

    topk_inds = topk_inds % (height * width)   # 目标在宽高上的索引  b*K
    topk_ys   = (topk_inds / width).int().float()   # 前K个目标的y坐标  b*K
    topk_xs   = (topk_inds % width).int().float()   # 前K个目标的x坐标  b*K
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs  # 前k个分值最大的目标的分值、在宽高上的索引、类别、y坐标、x坐标
```
</details>


### P3.6.5 crop_image

位于core\sample\utils.py

<details>

```python
def crop_image(image, center, size, output_size=None):
    if output_size == None:
        output_size = size

    cty, ctx            = center
    height, width       = size
    o_height, o_width   = output_size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((o_height, o_width, 3), dtype=image.dtype)

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = o_height // 2, o_width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - o_height // 2,
        ctx - o_width  // 2
    ])

    return cropped_image, border, offset
```
</details>
