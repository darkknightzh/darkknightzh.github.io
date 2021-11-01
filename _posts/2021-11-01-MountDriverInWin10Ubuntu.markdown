---
layout: post
title:  "windows10的ubuntu子系统挂载移动硬盘"
date:   2021-11-01 15:00:00 +0800
tags: [linux]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/MountDriverInWin10Ubuntu>


原始英文网址：

<https://linuxnightly.com/mount-and-access-hard-drives-in-windows-subsystem-for-linux-wsl/>

<br>

win10上安装ubuntu 20的子系统后，在/mnt目录下，能直接访问c，d等系统盘（不过win上大小写都行，ubuntu上只有小写才能访问）。但是若电脑连接了移动硬盘，则/mnt目录下有相应盘符（如电脑移动硬盘为E，则/mnt下有e目录，但直接进去是空文件夹），需要在ubuntu子系统中挂载该盘符。

说明：也可以通过下面的步骤挂载win上的虚拟光驱等设备。

① 在ubuntu中cd到/mnt目录：

```terminal
$ cd /mnt
```

② 如果需要挂载的移动硬盘盘符为E，则在/mnt中创建e文件夹（实际上不需要此步，因为/mnt下已经有了e的空文件夹）：

```terminal
$ sudo mkdir /mnt/e
```

③ 挂载移动硬盘到上述文件夹

```terminal
$ sudo mount -t drvfs E: /mnt/e
```

说明：E:为win上移动硬盘的盘符，后面不要加’/’，/mnt/e为需要挂载的路径。

④ 使用完之后，卸载该硬盘：

```terminal
$ sudo umount /mnt/e
```

⑤ 如需自动挂载相应文件，可将如下命令添加到ubuntu的/etc/fstab文件内（没使用过），不要改动该文件内已有的内容。

```text
E: /mnt/E drvfs defaults 0 0
```

⑥ 只读模式挂载，在之前命令中增加-o ro参数即可：

```terminal
$ sudo mount -o ro -t drvfs E: /mnt/e
```
