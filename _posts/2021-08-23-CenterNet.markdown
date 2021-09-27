---
layout: post
title:  "CenterNet Objects as Points"
date:   2021-08-23 16:00:00 +0800
tags: [deep learning, algorithm, detection, detection]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/CenterNet>

论文：

<https://arxiv.org/abs/1904.07850>

官方代码：

<https://github.com/xingyizhou/CenterNet>


## P1. 摘要
CenterNet为anchor-free的方法。如图1所示，本文将目标建模为其边界框的中心点。边界框的大小和其他属性可以通过目标中心关键点的特征推断。


![1](/assets/post/2021-08-23-CenterNet/1centernet.png)
_图1 centernet_

本文算法可以认为是一个单形不可知的anchor（single shape-agnostic anchor），如图2所示。优点：① CenterNet基于位置，而不是框的重叠。本文对于前景和背景分类，无需手动设定阈值。② 对于每个目标，本文只有一个正的“anchor”，因而不需要NMS。在特征图上直接提取局部峰值。③ CenterNet使用更大的输出分辨率（stride=4）。

![2](/assets/post/2021-08-23-CenterNet/2centerpixel.png)
_图2_


## P2. 预备知识

假定
$$I\in { {R}^{W\times H\times 3}}$$
为W*H的彩色图像。本文目标是得到关键点热图
$$\hat{Y}\in { {\left[ 0,1 \right]}^{\frac{W}{R}\times \frac{H}{R}\times C}}$$
，其中R为输入相比于输出的步长（stride）。如果是人体姿态估计，则C=17；如果是目标检测，则C=80（COCO数据库）。本文使用默认步长R=4，即输出分辨率为输入分辨率的1/4。预测的
$${ {\hat{Y}}_{x,y,c}}=1$$
为检测的关键点，
$${ {\hat{Y}}_{x,y,c}}=0$$
为背景。本文使用不同的网络来预测图像对应的
$${ {\hat{Y}}_{x,y,c}}$$
：堆叠hourglass网络（stacked hourglass network），ResNet，deep layer aggregation (DLA)。
对于每个类别为c的GT关键点
$$p\in { {R}^{2}}$$
，计算低分辨率的特征
$$\tilde{p}\in \left\lfloor \frac{p}{R} \right\rfloor $$
。

而后将GT关键点使用高斯核映
$${ {Y}_{xyc}}\in \exp \left( -\frac{ { {\left( x-{ { {\tilde{p}}}_{x}} \right)}^{2}}+{ {\left( y-{ { {\tilde{p}}}_{y}} \right)}^{2}}}{2\sigma _{p}^{2}} \right)$$
射到热图
$$Y\in { {\left[ 0,1 \right]}^{\frac{W}{R}\times \frac{H}{R}\times C}}$$
上，其中
$${ {\sigma }_{p}}$$
为目标尺寸自适应的标准差。若相同类别的的两个高斯核重叠，则使用两个高斯核中的更大值。训练时的损失函数为使用focal loss的逻辑回归：

$${ {L}_{k}}=\frac{-1}{N}\sum\limits_{xyc}{\left\{ \begin{matrix}
   { {\left( 1-{ { {\hat{Y}}}_{xyc}} \right)}^{\alpha }}\log \left( { { {\hat{Y}}}_{xyc}} \right) & if\text{ }{ {Y}_{xyc}}=1  \\
   { {\left( 1-{ {Y}_{xyc}} \right)}^{\beta }}\log { {\left( { { {\hat{Y}}}_{xyc}} \right)}^{\alpha }}\log \left( 1-{ { {\hat{Y}}}_{xyc}} \right) & otherwise  \\
\end{matrix} \right.} \tag{1}$$

其中
$$\alpha $$
和
$$\beta $$
为focal loss中的超参。N为图像I中的关键点数量。除以N是为了使所有正focal loss的实例归一化到1。本文使用
$$\alpha =2$$
，
$$\beta \text{=}4$$。

**说明**：公式1中，第一行，
$${ {Y}_{xyc}}$$
为1时：当预测值
$${ {\hat{Y}}_{xyc}}$$
接近1，第一项给予小的惩罚；否则给予大的惩罚。第二行，
$${ {Y}_{xyc}}$$
不为1时：①若
$${ {Y}_{xyc}}$$
在中心点周围，理论上
$${ {\hat{Y}}_{xyc}}$$
为0，实际上
$${ {\hat{Y}}_{xyc}}$$
接近1时，第二项给予大的惩罚，由于距离中心点较近，
$${ {\hat{Y}}_{xyc}}$$
接近1有可能，因而使用第一项降低一下惩罚；②若
$${ {Y}_{xyc}}$$
远离中心点，理论上
$${ {\hat{Y}}_{xyc}}$$
为0，实际上
$${ {\hat{Y}}_{xyc}}$$
接近1时，第二项给予大的惩罚，第一项保证距离中心越远的点的损失的权重越大，保证负样本检测。

由于对输出取整使用离散化，会带来预测误差，论文预测每个中心点的局部偏移，使用L1损失，偏移和类别无关，即只预测中心点的偏移，而不管是哪个类别的中心点，这样可以降低输出通道个数。L1损失如下：

$${ {L}_{off}}=\frac{1}{N}\sum\limits_{p}{\left| { { {\hat{O}}}_{ {\tilde{p}}}}-\left( \frac{p}{R}-\tilde{p} \right) \right|} \tag{2}$$

其中
$${ {\hat{O}}_{ {\tilde{p}}}}$$
为中心点预测坐标偏移，
$$\left( \frac{p}{R}-\tilde{p} \right)$$
为中心点实际坐标偏移。该L1损失只考虑关键点位置
$$\tilde{p}$$
，而不考虑其他位置。

**说明**：

① 不考虑具体类别时，偏移只考虑目标的x和y坐标，参数量：2\*目标个数；

② 考虑具体类别时，偏移需要考虑每个类别的目标的x和y坐标，参数量：2\*目标个数\*分类类别。


## P3. Objects as Points

假定
$$\left( x_{1}^{\left( k \right)},y_{1}^{\left( k \right)},x_{2}^{\left( k \right)},y_{2}^{\left( k \right)} \right)$$
为类别为
$${ {c}_{k}}$$
的目标k的边界框。其中心为
$$\left( \frac{x_{1}^{\left( k \right)}+x_{2}^{\left( k \right)}}{2},\frac{y_{1}^{\left( k \right)}+y_{2}^{\left( k \right)}}{2} \right)$$
，本文使用关键点预测器
$$\hat{Y}$$
来预测所有的中心点。另外，对每个目标k，还拟合目标大小
$${ {s}_{k}}=\left( x_{2}^{\left( k \right)}-x_{1}^{\left( k \right)},y_{2}^{\left( k \right)}-y_{1}^{\left( k \right)} \right)$$
。为了降低计算负担，对所有目标类别使用单独的预测器
$$\hat{S}\in { {R}^{\frac{W}{R}\times \frac{H}{R}\times 2}}$$
来预测目标宽高：

$${ {L}_{size}}=\frac{1}{N}\sum\limits_{k=1}^{N}{\left| { { {\hat{S}}}_{pk}}-{ {s}_{k}} \right|} \tag{3}$$

本文不缩放宽高，而是直接预测宽高的值。并且给预测宽高的损失较小的权重
$${ {\lambda }_{size}}$$
。总体的损失函数为：

$${ {L}_{\det }}\text{=}{ {L}_{k}}+{ {\lambda }_{size}}{ {L}_{size}}+{ {\lambda }_{off}}{ {L}_{off}} \tag{4}$$

其中
$${ {\lambda }_{size}}\text{=}0.1$$
，
$${ {\lambda }_{off}}\text{=}0.01$$
。本文使用一个网络来预测关键点
$$\hat{Y}$$
（目标中心），目标中心偏移
$$\hat{O}$$
和目标宽高
$$\hat{S}$$
。因而每个位置网络有C+4个输出。所有输出共享相同的骨干网络。骨干网络会通过独立的子网络得到每个预测信息，子网络为：3\*3 卷积+ReLU+1\*1 卷积。图3显示了网络输出。

![3](/assets/post/2021-08-23-CenterNet/3output.png)
_图3 output_

**从点到边界框**：推断阶段，首先提取每个类别的峰值（该点值大于等于其周围8个点的值作为峰值），并保留100个分值最高的峰值。令
$${ {\hat{P}}_{c}}$$
为n个检测到的类别为c的中心点集合：
$$\hat{P}=\left\{ \left( { { {\hat{x}}}_{i}},{ { {\hat{y}}}_{i}} \right) \right\}_{i=1}^{n}$$
。每个关键点位置为
$$\left( { {x}_{i}},{ {y}_{i}} \right)$$
。本文使用关键点的值
$${ {\hat{Y}}_{xiyic}}$$
作为其检测的置信度，从而得到检测框

$$\left( { { {\hat{x}}}_{i}}+\delta { { {\hat{x}}}_{i}}-{ { { {\hat{w}}}_{i}}}/{2}\;,\text{ }{ { {\hat{y}}}_{i}}+\delta { { {\hat{y}}}_{i}}-{ { { {\hat{h}}}_{i}}}/{2}\;,\text{ }{ { {\hat{x}}}_{i}}+\delta { { {\hat{x}}}_{i}}+{ { { {\hat{w}}}_{i}}}/{2,\text{ }{ { {\hat{y}}}_{i}}+\delta { { {\hat{y}}}_{i}}+{ { { {\hat{h}}}_{i}}}/{2}\;}\; \right)$$
&nbsp;

其中
$$\left( \delta { { {\hat{x}}}_{i}},\delta { { {\hat{y}}}_{i}} \right)={ {\hat{O}}_{ { { {\hat{x}}}_{i}},{ { {\hat{y}}}_{i}}}}$$
为预测的偏移，
$$\left( { { {\hat{w}}}_{i}},{ { {\hat{h}}}_{i}} \right)={ {\hat{S}}_{ { { {\hat{x}}}_{i}},{ { {\hat{y}}}_{i}}}}$$
为预测的宽高。所有的输出直接通过关键点估计，不需要NMS。关键点峰值提取可以代替NMS，并且可以使用3\*3 max pooling高效得到关键点峰值。

说明：可以将CenterNet应用于3D检测和人体姿态估计。具体请看论文。

**网络结构**：如图4所示

![4](/assets/post/2021-08-23-CenterNet/4model.png)
_图4 model diagrams_

## P4.代码

### P4.1 训练

训练代码位于main.py中

<details>

```python
def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)   # 得到自定义的dataset
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)   # 使用opt参数更新Dataset的相关参数
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)   # arch为dla_34等，heads为相关参数，head_conv为通道数量，最终得到相关模型
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)  # 载入之前模型

    Trainer = train_factory[opt.task]  
    trainer = Trainer(opt, model, optimizer)  # 得到训练迭代器，还是叫什么名字？？？
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)  # Trainer继承自BaseTrainer，里面有set_device

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    if opt.test:
        _, preds = trainer.val(0, val_loader)   # Trainer继承自BaseTrainer，里面有val
        val_loader.dataset.run_eval(preds, opt.save_dir)   # 测试结果
        return
    
    # 得到loader，比如CTDetDataset，里面有__getitem__，可以取数据
    train_loader = torch.utils.data.DataLoader(Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)   # Trainer继承自BaseTrainer，里面有train，训练一个epoch
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)  # 保存模型
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)  # 测试训练一个epoch后的结果
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:  # opt.metric=loss，此处指损失更低
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:   # 保存模型，更新lr
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
```
</details>

### P4.2 测试

测试代码位于test.py中

<details>

```python
class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:   # 缩放图像
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)

def prefetch_test(opt):   #  对coco图像缩放，并测试最终结果
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(PrefetchDataset(opt, dataset, detector.pre_process), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)
        results[img_id.numpy().astype(np.int32)[0]] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(t, tm=avg_time_stats[t])
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)

def test(opt):   # 测试coco的原始分辨率图像
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset] 
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]     # 得到相应的检测器，如CtdetDetector

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):   # 测试的时候，batchsize=1
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        if opt.task == 'ddd':
            ret = detector.run(img_path, img_info['calib'])
        else:
            ret = detector.run(img_path)   # 调用CtdetDetector的基类BaseDetector的run函数

        results[img_id] = ret['results']

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
    opt = opts().parse()
    if opt.not_prefetch_test:  
        test(opt)   # 测试coco的原始分辨率图像
    else:
        prefetch_test(opt)  # 对coco图像缩放，并测试最终结果
```
</details>

### P4.3 demo

<details>

```python
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)   # 得到相应的检测器，如CtdetDetector

    if opt.demo == 'webcam' or opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:   # webcam或者视频
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):  # 图像文件夹
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]   # 图像文件

        for (image_name) in image_names:
            ret = detector.run(image_name)    # 调用CtdetDetector的基类BaseDetector的run函数
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
```
</details>


### P4.4 默认参数opt

位于lib/opts.py

<details>

```python
class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('task', default='ctdet', help='ctdet | ddd | multi_pose | exdet')
        self.parser.add_argument('--dataset', default='coco', help='coco | kitti | coco_hp | pascal')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--debug', type=int, default=0, help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplot to display'  # useful when lunching training with ipython notebook
                                      '4: save all visualizations to disk')
        self.parser.add_argument('--demo', default='', help='path to image/ image folders/ video. or "webcam"')
        self.parser.add_argument('--load_model', default='', help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true', help='resume an experiment. '
                                      'Reloaded the optimizer parameter and set load_model to model_last.pth in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4, help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317, help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--print_iter', type=int, default=0, help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true', help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true', help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss', help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3, help='visualization threshold.')   # 显示检测结果的阈值
        self.parser.add_argument('--debugger_theme', default='white', choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', default='dla_34', help='model architecture. Currently tested res_18 | res_101 | resdcn_18 | resdcn_101 | dlav0_34 | dla_34 | hourglass')
        self.parser.add_argument('--head_conv', type=int, default=-1, help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4, help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1, help='input height and width. -1 for default from dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1, help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1, help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4, help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120', help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140, help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1, help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1, help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true', help='include validation in training and test on test set')

        # test
        self.parser.add_argument('--flip_test', action='store_true', help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1', help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true', help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100, help='max number of output objects.')   # 做多检测的目标数量
        self.parser.add_argument('--not_prefetch_test', action='store_true', help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true', help='fix testing resolution or keep the original resolution')
        self.parser.add_argument('--keep_res', action='store_true', help='keep the original resolution during validation.')

        # dataset
        self.parser.add_argument('--not_rand_crop', action='store_true', help='not use the random crop data augmentation from CornerNet.')
        self.parser.add_argument('--shift', type=float, default=0.1, help='when not using random crop apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0.4, help='when not using random crop apply scale augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0, help='when not using random crop apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.5, help='probability of applying flip augmentation.')
        self.parser.add_argument('--no_color_aug', action='store_true', help='not use the color augmenation from CornerNet')
        # multi_pose
        self.parser.add_argument('--aug_rot', type=float, default=0, help='probability of applying rotation augmentation.')
        # ddd
        self.parser.add_argument('--aug_ddd', type=float, default=0.5, help='probability of applying crop augmentation.')
        self.parser.add_argument('--rect_mask', action='store_true', help='for ignored object, apply mask on the rectangular region or just center point.')
        self.parser.add_argument('--kitti_split', default='3dop', help='different validation split for kitti: 3dop | subcnn')

        # loss
        self.parser.add_argument('--mse_loss', action='store_true', help='use mse loss or focal loss to train keypoint heatmaps.')   # 为False，否则lib\datasets\sample\ctdet.py中hm_gauss未定义
        # ctdet
        self.parser.add_argument('--reg_loss', default='l1', help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1, help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1, help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1, help='loss weight for bounding box size.')
        # multi_pose
        self.parser.add_argument('--hp_weight', type=float, default=1, help='loss weight for human pose offset.')
        self.parser.add_argument('--hm_hp_weight', type=float, default=1, help='loss weight for human keypoint heatmap.')
        # ddd
        self.parser.add_argument('--dep_weight', type=float, default=1, help='loss weight for depth.')
        self.parser.add_argument('--dim_weight', type=float, default=1, help='loss weight for 3d bounding box size.')
        self.parser.add_argument('--rot_weight', type=float, default=1, help='loss weight for orientation.')
        self.parser.add_argument('--peak_thresh', type=float, default=0.2)

        # task
        # ctdet
        self.parser.add_argument('--norm_wh', action='store_true', help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true', help='apply weighted regression near center or just apply regression on center point.')  # 目标中心是否加权
        self.parser.add_argument('--cat_spec_wh', action='store_true', help='category specific bounding box size.')   # 特定类别的box size
        self.parser.add_argument('--not_reg_offset', action='store_true', help='not regress local offset.')
        # exdet
        self.parser.add_argument('--agnostic_ex', action='store_true', help='use category agnostic extreme points.')
        self.parser.add_argument('--scores_thresh', type=float, default=0.1, help='threshold for extreme point heatmap.')
        self.parser.add_argument('--center_thresh', type=float, default=0.1, help='threshold for centermap.')
        self.parser.add_argument('--aggr_weight', type=float, default=0.0, help='edge aggregation weight.')
        # multi_pose
        self.parser.add_argument('--dense_hp', action='store_true', help='apply weighted pose regression near center '
                                      'or just apply regression on center point.')
        self.parser.add_argument('--not_hm_hp', action='store_true', help='not estimate human joint heatmap, directly use the joint offset from center.')
        self.parser.add_argument('--not_reg_hp_offset', action='store_true', help='not regress local offset for human joint heatmaps.')
        self.parser.add_argument('--not_reg_bbox', action='store_true', help='not regression bounding box size.')

        # ground truth validation
        self.parser.add_argument('--eval_oracle_hm', action='store_true', help='use ground center heatmap.')
        self.parser.add_argument('--eval_oracle_wh', action='store_true', help='use ground truth bounding box size.')
        self.parser.add_argument('--eval_oracle_offset', action='store_true', help='use ground truth local heatmap offset.')
        self.parser.add_argument('--eval_oracle_kps', action='store_true', help='use ground truth human pose offset.')
        self.parser.add_argument('--eval_oracle_hmhp', action='store_true', help='use ground truth human joint heatmaps.')
        self.parser.add_argument('--eval_oracle_hp_offset', action='store_true', help='use ground truth human joint local offset.')
        self.parser.add_argument('--eval_oracle_dep', action='store_true', help='use ground truth depth.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset   # 约束局部偏移
        opt.reg_bbox = not opt.not_reg_bbox
        opt.hm_hp = not opt.not_hm_hp
        opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):  # 更新dataset，设置检测的head
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h   # 更新默认值
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio   # down_ratio默认为4
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'exdet':
            # assert opt.dataset in ['coco']
            num_hm = 1 if opt.agnostic_ex else opt.num_classes
            opt.heads = {'hm_t': num_hm, 'hm_l': num_hm, 'hm_b': num_hm, 'hm_r': num_hm, 'hm_c': opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
        elif opt.task == 'ddd':
            # assert opt.dataset in ['gta', 'kitti', 'viper']
            opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
            if opt.reg_bbox:
                opt.heads.update({'wh': 2})
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        elif opt.task == 'ctdet':    # 检测
            # assert opt.dataset in ['pascal', 'coco']
            opt.heads = {'hm': opt.num_classes, 'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}  # hm 热图数量   wh：不限定特定类别box大小的话，就是2。否则，就是2*类别数量
            if opt.reg_offset:   # 约束局部偏移
                opt.heads.update({'reg': 2})
        elif opt.task == 'multi_pose':
            # assert opt.dataset in ['coco_hp']
            opt.flip_idx = dataset.flip_idx
            opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            if opt.hm_hp:
                opt.heads.update({'hm_hp': 17})
            if opt.reg_hp_offset:
                opt.heads.update({'hp_offset': 2})
        else:
            assert 0, 'task not defined!'
        print('heads', opt.heads)
        return opt

    def init(self, args=''):
        default_dataset_info = {   # 默认dataset的信息
            'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278], 'dataset': 'coco'},
            'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278], 'dataset': 'coco'},
            'multi_pose': {'default_resolution': [512, 512], 'num_classes': 1, 'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'coco_hp', 'num_joints': 17, 'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]},
            'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'dataset': 'kitti'},
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)
        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
```
</details>

### P4.5 模型相关

#### P4.5.1 create_model

均位于lib/models/model.py

<details>

```python
_model_factory = {
    'res': get_pose_net,  # default Resnet with deconv  含有deconv的resnet
    'dlav0': get_dlav0,  # default DLAup
    'dla': get_dla_dcn,
    'resdcn': get_pose_net_dcn,
    'hourglass': get_large_hourglass_net,
}

def create_model(arch, heads, head_conv):   # arch为dla_34等，heads为相关参数，head_conv为通道数量
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]   # 得到实际模型的名字
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)   # 得到实际模型
return model
```
</details>

#### P4.5.2 load_model

均位于lib/models/model.py

<details>

```python
def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model
```
</details>

#### P4.5.3 save_model

均位于lib/models/model.py

<details>

```python
def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
```
</details>

#### 4.5.4 get_pose_net

位于lib/models/networks/msra_resnet.py

<details>

```python
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):   # ((x or downsample(x))  +  (conv+bn+relu  +  conv+bn))  +  relu
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x): 
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):   # ((x or downsample(x))  +  (conv+bn+relu  +  conv+bn+relu  +  conv+bn))  +  relu
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x): 
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # 输出特征宽高为图像宽高的1/32

        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4], )   # 输出特征宽高为图像宽高的1/4    # used for deconv layers
        # self.final_layer = []

        for head in sorted(self.heads):  # {'hm': num_classes, 'reg': 2, 'wh': 2}
            num_output = self.heads[head]  # 分别得到num_classes,2,2
            if head_conv > 0:   # head_conv=256
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))  # 最后的1*1 conv作为分类层，得到num_output维度的特征
            else:
                fc = nn.Conv2d(in_channels=256, out_channels=num_output, kernel_size=1, stride=1, padding=0)   # 直接使用1*1 conv作为分类层，得到num_output维度的特征
            self.__setattr__(head, fc)   # 设置hm，reg，wh这些层

        # self.final_layer = nn.ModuleList(self.final_layer)

    def _make_layer(self, block, planes, blocks, stride=1):  # block：BasicBlock or Bottleneck
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):   # num_layers=3
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)   # num_kernels=[4, 4, 4]   输出4,1,0

            planes = num_filters[i]   # num_filters=[256, 256, 256]
            # 转置卷积，增加输出特征分辨率
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias)) 
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:   # hm，reg，wh这些层
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        if m.weight.shape[0] == self.heads[head]:
                            if 'hm' in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.normal_(m.weight, std=0.001)
                                nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_pose_net(num_layers, heads, head_conv):  # head_conv=256
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers, pretrained=True)
    return model
```
</details>

### P4.6 得到数据get_dataset

得到数据数据，供dataloader使用的get_dataset位于src/lib/datasets/dataset_factory.py。其根据输入得到实际的Dataset。

#### P4.6.1 get_dataset

<details>

```python
dataset_factory = {
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP
}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset
}

def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):   # 自定义的dataset
        pass
    return Dataset
```
</details>


#### P4.6.2 CTDetDataset

其位于src/lib/datasets/sample/ctdet.py。

<details>

```python
class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):  # 从coco的[x_start,y_start_w,h]变换到bbox的[x1,y1,x2,y2]
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):      # 以下均使用COCO做注释，如果使用lib\datasets\dataset_factory.py中其他类（如PascalVOC、KITTI），应该类似
        img_id = self.images[index]   # 得到图像id
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']   # loadImgs为代码中类的COCO的self.coco的函数，返回list，因而使用[0]，最终得到文件名字
        img_path = os.path.join(self.img_dir, file_name)   # self.img_dir为代码中类的COCO自带的变量
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])   # getAnnIds为代码中类的COCO的self.coco的函数
        anns = self.coco.loadAnns(ids=ann_ids)          # loadAnnsgetAnnIds为代码中类的COCO的self.coco的函数
        num_objs = min(len(anns), self.max_objs)        # 目标数量的最小值

        img = cv2.imread(img_path)  # 读取图像

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)   # 图像中心，即仿射变换的中心
        if self.opt.keep_res:    # 验证时是否保持图像原始分辨率
            input_h = (height | self.opt.pad) + 1   # pad = 127 if 'hourglass' in opt.arch else 31   此处为按位或，+1后得到2的整数倍（与height及pad实际数值有关）
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)   # 仿射变换输入宽高
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0    # 宽高较大者
            input_h, input_w = self.opt.input_h, self.opt.input_w   # 输入宽高

        flipped = False
        if self.split == 'train':   # 训练集   # 下面需要使用的是c（仿射变换的中心）、s（仿射变换的缩放倍数）
            if not self.opt.not_rand_crop:   # not self.opt.not_rand_crop默认True
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))   # 随机缩放一定倍数
                w_border = self._get_border(128, img.shape[1])    # 仿射变换使用，不懂为何这样计算？？？
                h_border = self._get_border(128, img.shape[0])    # 仿射变换使用，不懂为何这样计算？？？
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)   # 随机选一个值
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:   # 下面需要使用的是c（仿射变换的中心）、s（仿射变换的缩放倍数）
                sf = self.opt.scale   # 0.4
                cf = self.opt.shift   # 0.1
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:   # 0.5的概率左右翻转图像
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])  # 得到仿射变换的矩阵
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)  # 对图像进行仿射变换
        inp = (inp.astype(np.float32) / 255.)   # 图像缩放到0-1之间
        if self.split == 'train' and not self.opt.no_color_aug: 
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)    # 对图像进行色彩扰动，self._data_rng为初始化种子的state
        inp = (inp - self.mean) / self.std   # 归一化
        inp = inp.transpose(2, 0, 1)       # HWC转CHW

        output_h = input_h // self.opt.down_ratio    # down_ratio=4
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])   # 输出的仿射变换矩阵，用于变换相应bbox

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)   # 热图数量
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)             # 每个目标的wh（供预测的wh和真实目标的wh计算损失）  max_objs*2
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)  # 目标权重矩阵（中心和非中心）
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)            # 每个目标中心的偏移   max_objs*2
        ind = np.zeros((self.max_objs), dtype=np.int64)                 # 输出特征图上每个目标中心的索引：y*w+hT    max_objs
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)            # 每个目标的mask：1代表有目标，0代表没目标（默认）   max_objs
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)   # 每个目标包含label的wh     max_objs*(num_classes*2)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)   # 每个目标包含label的mask   max_objs*(num_classes*2)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian   # 使用mse 损失，则为draw_msra_gaussian；使用focal loss，则为draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):  # 依次遍历每个目标
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])   # 得到bbox
            cls_id = int(self.cat_ids[ann['category_id']])   # 得到目标类别
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1   # 水平反转目标信息
            bbox[:2] = affine_transform(bbox[:2], trans_output)     # 计算仿射变换后的xy
            bbox[2:] = affine_transform(bbox[2:], trans_output)     # 计算仿射变换后的wh
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)   # 裁剪bbox
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]             # 最终目标宽高
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))   # 得到目标的半径？？？
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius   # 无hm_gauss，因而opt.mse_loss为False，另外使用Focal loss计算热图的损失
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  # 目标中心
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)  # 根据目标类别、目标中心、半径画高斯热图
                wh[k] = 1. * w, 1. * h   # 第k个目标的wh的值设置为相应目标的宽高
                ind[k] = ct_int[1] * output_w + ct_int[0]   # y*w+h，得到第k个ind的值
                reg[k] = ct - ct_int       # 第k个目标中心的偏移
                reg_mask[k] = 1       # mask为1，代表有目标
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]   # 第k个目标包含label的wh的值设置为相应目标的宽高
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1     # 第k个目标包含label的mask的值设置为1
                if self.opt.dense_wh:    # 目标中心是否加权（True是加权）
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)   # 更新dense_wh（目标中心和边缘的权重矩阵）
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])   # 实际目标box  [x1,y1,x2,y2]

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}  # 返回值（图像，热图，是否有目标的mask，每个目标中心的索引，每个目标的wh）

        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)   # 热图最大值
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)  # 2*output_h*output_h
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})  # 更新相关返回值
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']

        if self.opt.reg_offset:  # 更新目标中心的偏移offset
            ret.update({'reg': reg})

        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        
        return ret
```
</details>

#### P4.6.3 color_aug

位于src/lib/utils/image.py。

<details>

```python
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)   # 不懂这个为什么，光照？？？

def blend_(alpha, image1, image2):   # 对图像加权融合
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])  # 对图像加权融合

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha   # 直接乘系数

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)   # 对图像加权融合

def color_aug(data_rng, image, eig_val, eig_vec):  # _data_rng为初始化种子的state
    functions = [brightness_, contrast_, saturation_]   # 亮度，对比度，饱和度
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)   # 调整亮度
```
</details>


#### P4.6.4 draw_msra_gaussian

位于src/lib/utils/image.py

<details>

```python
def draw_msra_gaussian(heatmap, center, sigma):   # sigma为radius
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]   # 左上角
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   # 右下角
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2   # 临时高斯分布g的中心
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))   # 高斯分布的临时变量g
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]   # -ul[0]是减去各自的偏移，得到g上的xy坐标（各自的起点和终点）  
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)     # 热图上额xy坐标（各自的起点和终点）
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])   # 得到以center为中心高斯分布的热图
    return heatmap
```
</details>

#### P4.6.5 draw_umich_gaussian

位于src/lib/utils/image.py

<details>

```python
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]   # y：k1*1， x：1：k2    和meshgrid的区别是，meshgrid两个返回值都是k1*k2的

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))   # 得到高斯分布的矩阵   k1*k2
    h[h < np.finfo(h.dtype).eps * h.max()] = 0   # 将太小的设置为0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
```
</details>


#### P4.6.6 draw_dense_reg

位于src/lib/utils/image.py

<details>

```python
def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)   # 得到高斯分布的矩阵 
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)   # value为第k个目标的wh（[2]的向量）    2*1*1
    dim = value.shape[0]   # 2
    reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value   # 第k个目标的宽高的mask
    if is_offset and dim == 2:   # 默认不执行
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]    # heatmap为height*width的矩阵（原始热图不同目标取最大值）

    left, right = min(x, radius), min(width - x, radius + 1)   # 以center为中心，radius为半径的框（圆转换成正方形）的边界
    top, bottom = min(y, radius), min(height - y, radius + 1)  # 以center为中心，radius为半径的框（圆转换成正方形）的边界

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]   # heatmap上该区域的mask
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]  # regmap上该区域的mask
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]   # gaussian上该区域的mask
    masked_reg = reg[:, radius - top:radius + bottom, radius - left:radius + right]          # reg上该区域的mask
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(1, masked_gaussian.shape[0], masked_gaussian.shape[1])   # 目标区域目标权重>热图权重的索引
        masked_regmap = (1-idx) * masked_regmap + idx * masked_reg   # 目标区域目标权重<热图的位置，使用原来的权重；否则使用新的权重？？？
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap   # 更新相应位置的权重
    return regmap
```
</details>

#### P4.6.7 gaussian_radius

位于src/lib/utils/image.py

<details>

```python
def gaussian_radius(det_size, min_overlap=0.7):   # 得到目标的半径？？？不清楚下面计算步骤
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)
```
</details>

### P4.7 训练迭代器train_factory

位于src/lib/trains/train_factory.py。

4.7.1 train_factory

<details>

```python
train_factory = {
    'exdet': ExdetTrainer,
    'ddd': DddTrainer,
    'ctdet': CtdetTrainer,
    'multi_pose': MultiPoseTrainer,
}
```
</details>

#### P4.7.2 CtdetTrainer

位于src/lib/trains/ctdet.py。

<details>

```python
class CtdetTrainer(BaseTrainer):  # 继承自BaseTrainer，里面包含相应的训练代码（前向，loss反向等）
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None   # 目标中心偏移的输出
        dets = ctdet_decode(output['hm'], output['wh'], reg=reg, cat_spec_wh=opt.cat_spec_wh, K=opt.K)  # k为最多检测的目标数量
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1], dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1], dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(output['hm'], output['wh'], reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
print("Hello, World!");
```
</details>

#### P4.7.3 BaseTrainer

CtdetTrainer继承自BaseTrainer。BaseTrainer位于src/lib/trains/base_trainer.py。

<details>

```python
class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])  # 得到模型输出
        loss, loss_stats = self.loss(outputs, batch)  # 计算模型损失
        return outputs[-1], loss, loss_stats   # 返回模型输出，总损失，各自损失

class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)  # 派生类（如CtdetTrainer）有相应的实现
        self.model_with_loss = ModelWithLoss(model, self.loss)  # 得到模型输出，总损失，各自损失

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(self.model_with_loss, device_ids=gpus, chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss   # 得到模型输出，总损失，各自损失
        if phase == 'train':  # 训练阶段
            model_with_loss.train()
        else:   # 测试阶段
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)   # 得到实际的模型输出，总损失，各自损失
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(epoch, iter_id, num_iters, phase=phase, total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) | Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug > 0:
                self.debug(batch, output, iter_id)  # 调试结果

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError   # 派生类（如CtdetTrainer）有相应的实现

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError  # 派生类（如CtdetTrainer）有相应的实现

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
print("Hello, World!");
```
</details>


#### P4.7.4 后处理ctdet_post_process

位于src/lib/utils/post_process.py。

<details>

```python
def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):   # 依次遍历batch中每个结果
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))  # 得到预测目标对应的原始图像上的xy坐标
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h)) # 得到预测目标对应的原始图像上的wh
        classes = dets[i, :, -1]   # 得到预测的类别
        for j in range(num_classes):   # 得到每个类别的预测信息（bbox、分值），其中0为背景
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([dets[i, inds, :4].astype(np.float32), dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()   # 当前结果转成list（后面还会转成np。。。）
        ret.append(top_preds)
    return ret  # 得到当前batch中每个图像各类别的预测信息（bbox、分值）
print("Hello, World!");
```
</details>

#### P4.7.5 transform_preds

ctdet_post_process调用transform_preds，其位于src/lib/utils/image.py

<details>

```python
def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)   # 得到目标矩阵向原始矩阵的仿射变换矩阵
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)  # 计算仿射变换后的坐标，得到预测目标对应的原始图像上的坐标
    return target_coords

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):   # 得到仿射变换矩阵
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))   # 目标矩阵向原始矩阵的仿射变换矩阵
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))   # 原始矩阵向目标矩阵的仿射变换矩阵

    return trans

def affine_transform(pt, t):   # 计算仿射变换后的坐标
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
```
</details>

#### P4.7.6 ctdet_decode

位于src/lib/models/decode.py

<details>

```python
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()  # 热图和对热图maxpool之后的值相等的点，是局部极大值点。
    return heat * keep   # 得到极大值点矩阵

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)   # 得到当前batch中不同通道的总共K个极值点，用于得到每个类别极值点（即目标中心）的坐标  b*C*K

    topk_inds = topk_inds % (height * width)       # 每个类别极值点的yx坐标  b*C*K
    topk_ys = (topk_inds / width).int().float()    # b*C*K
    topk_xs = (topk_inds % width).int().float()    # b*C*K

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)   # 用于得到极值点（即目标）的得分及类别索引  b*K
    topk_clses = (topk_ind / K).int()     #  极值点（即目标）的类别

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)   # _gather_feat得到ind指定的点的channel上的特征，此处得到前k个目标的索引
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)    # 得到前k个目标的y坐标
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)   # 得到前k个目标的x坐标

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs   # 分值最高的前k个分值，对应的索引(w*i+j)，对应的类别，对应的y坐标，对应的x坐标

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)   # 得到极大值点矩阵（和热图shape一样）

    scores, inds, clses, ys, xs = _topk(heat, K=K)    # 分值最高的前k个分值，对应的索引(w*i+j)，对应的类别，对应的y坐标，对应的x坐标
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)   # 得到当前batch中每个feat相应索引处的2维特征，此处得到前k个目标预测坐标的偏移
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]    # 极值点的整数坐标+预测偏移，得到目标的预测坐标的中心
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)   # 得到预测的目标宽高
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2, xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)  # 预测的bbox
    detections = torch.cat([bboxes, scores, clses], dim=2)   # b*K*6

    return detections   # 得到预测的bbox、分值、类别
```
</details>

### P4.8 损失函数

CtdetTrainer调用CtdetLoss。其位于src/lib/trains/ctdet.py。

#### P4.8.1 CtdetLoss

<details>

```python
class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()  # 使用MSe loss或者focal loss训练关键点热图的损失，实际使用的是focal loss
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else RegLoss() if opt.reg_loss == 'sl1' else None  # 训练回归的损失
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else NormRegL1Loss() if opt.norm_wh else RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg  # 预测目标宽高的损失
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(opt.num_stacks):   # 模型为hourglass时num_stacks=2。其他情况为1
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:  # use ground center heatmap
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:  # use ground truth bounding box size
                output['wh'] = torch.from_numpy(gen_oracle_map(batch['wh'].detach().cpu().numpy(), batch['ind'].detach().cpu().numpy(), output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:  # use ground truth local heatmap offset
                output['reg'] = torch.from_numpy(gen_oracle_map(batch['reg'].detach().cpu().numpy(), batch['ind'].detach().cpu().numpy(), output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks  # 热图的损失
            if opt.wh_weight > 0:  # 默认0.1  计算预测wh的损失
                if opt.dense_wh:    # torch.nn.L1Loss   预测值和真值的绝对值的平均值
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4   # dense_wh_mask为2个热图的最大值concatenate而成：2*output_h*output_h  dense_wh:目标中心和边缘的权重矩阵
                    wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'], batch['dense_wh'] * batch['dense_wh_mask']) / mask_weight) / opt.num_stacks   # 模型为hourglass时num_stacks=2。其他情况为1
                elif opt.cat_spec_wh:  # RegWeightedL1Loss  实际上还是L1 loss   output['wh']：B*2*H*W
                    wh_loss += self.crit_wh(output['wh'], batch['cat_spec_mask'], batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:  # NormRegL1Loss(l1_loss(pred/targe1, 1))或者RegL1Loss(l1_loss(pred, targe1))
                    wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:   # off_weight默认1，reg_offset默认True  计算预测wh偏移的损失
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss   # 总的损失
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}  # 各自的损失
        return loss, loss_stats
```
</details>

#### P4.8.2 FocalLoss

位于src/lib/models/losses.py。

<details>

```python
def _neg_loss(pred, gt):  # focal loss，通过class FocalLoss调用
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)
```
</details>

#### P4.8.3 RegLoss

位于src/lib/models/losses.py。

<details>

```python
def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss
```
</details>

#### P4.8.4 RegL1Loss

位于src/lib/models/losses.py。

<details>

```python
class RegL1Loss(nn.Module):    # l1_loss(pred, targe1)
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target): 
        pred = _transpose_and_gather_feat(output, ind)   # 得到当前batch中每个feat相应索引处的2维特征，不知道ind在哪里转的batch？？？   B*k*2
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss
```
</details>

#### P4.8.5 NormRegL1Loss

位于src/lib/models/losses.py。

<details>

```python
class NormRegL1Loss(nn.Module):    # l1_loss(pred/targe1, 1)
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target): 
        pred = _transpose_and_gather_feat(output, ind)    # 得到当前batch中每个feat相应索引处的2维特征，不知道ind在哪里转的batch？？？   B*k*2
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)    
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss
```
</details>

#### P4.8.6 RegWeightedL1Loss

位于src/lib/models/losses.py。

<details>

```python
class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):   # output：B*2*H*W    mask：max_objs*(num_classes*2)    target：max_objs*(num_classes*2)
        pred = _transpose_and_gather_feat(output, ind)  # 得到当前batch中每个feat相应索引处的2维特征，不知道ind在哪里转的batch？？？   B*k*2
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss    # 得到损失
```
</details>


#### P4.8.7 其他函数

位于src/lib/models/utils.py

<details>

```python
def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):  # 得到ind指定的点的channel上的特征  ind是2维（不知代码中什么地方加上的batch）
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)       # 依次得到当前batch中每个feat相应索引处的2维特征  B*k*2
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):  # feat：B*2*H*W
    feat = feat.permute(0, 2, 3, 1).contiguous()   # BCHW变成BHWC
    feat = feat.view(feat.size(0), -1, feat.size(3))   # BHWC变成B*(HW)*C
    feat = _gather_feat(feat, ind)  # 得到当前batch中每个feat相应索引处的2维特征   B*k*2
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)
```
</details>

### P4.9 测试时的检测类detector_factory

位于src/lib/detectors/detector_factory.py。

#### P4.9.1 detector_factory

<details>

```python
detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector, 
}
```
</details>

#### P4.9.2 CtdetDetector

位于src/lib/detectors/ctdet.py

<details>

```python
class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]  # 得到模型输出，默认输出为[]，因而取[-1]得到实际输出
            hm = output['hm'].sigmoid_()   # 预测的目标热图
            wh = output['wh']              # 预测的目标宽高
            reg = output['reg'] if self.opt.reg_offset else None   # 预测的目标偏移
            if self.opt.flip_test:   # 若flip_test为True，则输入网络的batchsize为2，否则为1
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2    # flip_tensor：torch.flip(x, [3])   hm[1]的为水平反转的图像的结果，因而将hm[1]也水平反转，使用[0:1]是为了保持batch维度
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)  # 得到预测的bbox、分值、类别

        if return_time:
            return output, dets, forward_time  # 返回模型输出，预测信息，前向时间
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], self.opt.num_classes) # 得到当前batch中每个图像各类别的预测信息（bbox、分值），其中0为背景
        for j in range(1, self.num_classes + 1):   # j从1开始，去掉背景
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)   # 之前转成list的结果在转会np。。。
            dets[0][j][:, :4] /= scale
        return dets[0]   # 测试的时候，bs为1，因而返回第0个结果，即当前图像各类别的预测信息（bbox、分值），其中0为背景

    def merge_outputs(self, detections):   # detections：当前图像各缩放尺度的各类别的预测信息（bbox、分值）
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)   # 当前类别在不同尺度上的结果拼接起来
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)   # 使用soft nms处理第j个类别的结果（对iou进行指数衰减，作为权重给后面的框的分值加权）
        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])  # 检测到的所有目标的分值
        if len(scores) > self.max_per_image:   # 超出检测的最大数量
            kth = len(scores) - self.max_per_image  # 需要去除的检测目标数量
            thresh = np.partition(scores, kth)[kth]  # np.partition将scores[kth]作为阈值，小于该值的都在kth前面，大于该值的都在kth后面。thresh为该阈值
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)   # 得到每个类大于该阈值的索引
                results[j] = results[j][keep_inds]      # 保留每个类超过该阈值的结果
        return results   # 融合不同尺度各类别的预测信息，并返回各类别的预测信息（bbox、分值）

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1], detection[i, k, 4], img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)
```
</details>

#### P4.9.3 BaseDetector

CtdetDetector继承自BaseDetector。BaseDetector位于src/lib/detectors/base_detector.py

<details>

```python
class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales   # 缩放图像，进行检测
        self.opt = opt
        self.pause = True

    def pre_process(self, image, scale, meta=None):  # 仿射变换，归一化，转到CHW
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 'out_height': inp_height // self.opt.down_ratio, 'out_width': inp_width // self.opt.down_ratio}
        return images, meta  # 返回图像及缩放相关信息

    def process(self, images, return_time=False):   # 调用派生类的process
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):   # 调用派生类的post_process
        raise NotImplementedError

    def merge_outputs(self, detections):   # 调用派生类的merge_outputs
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):     # 调用派生类的debug
        raise NotImplementedError

    def show_results(self, debugger, image, results):     # 调用派生类的show_results
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3), theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False   # 是否已经预处理过了
        if isinstance(image_or_path_or_tensor, np.ndarray):   # 输入为tensor
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):       # 输入为list
            image = cv2.imread(image_or_path_or_tensor)
        else:                                                 # 输入为dict
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True    # 为dict的话，应该是dataloader送进来的，已经预处理过了

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []   # 检测结果
        for scale in self.scales:   # 依次检测缩放的图像
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)   # 预处理图像, 返回图像及缩放相关信息
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images, return_time=True)   # 返回模型输出，预测信息，前向时间

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets, meta, scale)   # 调用派生类，如CtdetDetector的post_process，得到当前图像各类别的预测信息（bbox、分值），其中0为背景
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)   # 得到当前图像各缩放尺度的各类别的预测信息（bbox、分值）

        results = self.merge_outputs(detections)   # 融合不同尺度各类别的预测信息，并返回各类别的预测信息（bbox、分值）
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)

        return {'results': results, 'tot': tot_time, 'load': load_time, 'pre': pre_time, 'net': net_time, 'dec': dec_time, 'post': post_time, 'merge': merge_time}  # 返回相关结果
```
</details>

### P4.10 NMS

位于src/lib/external/nms.pyx，包括nms和soft_nms（还有其他的nms，没看）

<details>

```python
cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

def nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):  # 原始的nms方式，先对分值降序排序（实际是得到索引order），然后依次抑制
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]   # 对分值进行排序

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep

def soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov

    for i in range(N):   # 检测到的当前类别目标总数
        maxscore = boxes[i, 4]  # 取出第i个分值作为最大的分值
        maxpos = i  # 最大分值的位置

        tx1 = boxes[i,0]  # 坐标和分值信息
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1   # 依次遍历后面的目标
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]   # 得到从i开始的最大分值及位置
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]  # 和下段一起，交换i和maxpos的信息
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1  # 和上段一起，交换i和maxpos的信息
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]  # 更新当前位置的信息
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:   # 依次遍历后面（pos处）的目标
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 遍历的目标的面积
            iw = (min(tx2, x2) - max(tx1, x1) + 1)   # 当前框和pos处框的交集的宽
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)  # 当前框和pos处框的交集的高
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)   # 并集的面积
                    ov = iw * ih / ua #iou between max box and detection box    IoU

                    if method == 1: # linear
                        if ov > Nt:  
                            weight = 1 - ov    # IoU较大，给与较小的权重
                        else:
                            weight = 1         # IoU较小，给与权重为1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)   # 指数衰减方式得到权重
                    else: # original NMS
                        if ov > Nt:   # 原始方式，IoU大于阈值，则抑制pos处的相应框（保留当前框），否则保留pos处的相应框
                            weight = 0    # 抑制pos处的相应框
                        else:
                            weight = 1   # 保留pos处的相应框

                    boxes[pos, 4] = weight*boxes[pos, 4]   # 更新pos处框的分值
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:  # pos处框的分值小于阈值，则把最后一个box的信息更新到pos处，同时减少框的数量，并更新pos
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1   # 减少框的数量
                        pos = pos - 1  # 因为都在while循环里面，替换boxes[N-1]到boxes[pos]后，需要重新比较pos处的框，因而此处pos-1，下面pos+1后，重新比较pos处的框

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep
```
</details>
