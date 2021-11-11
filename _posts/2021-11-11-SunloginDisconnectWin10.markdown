---
layout: post
title:  "windows10使用向日葵访问ubuntu 20.04连接已断开"
date:   2021-11-11 14:50:00 +0800
tags: [linux]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>


参考网址：

<https://zhuanlan.zhihu.com/p/281051791>

<https://www.cnblogs.com/xlpc/p/12345478.html>


<br>

<https://zhuanlan.zhihu.com/p/281051791>指出：

```terminal
sudo apt update
sudo apt upgrade
sudo apt install lightdm
```

选择lightdm即可，需要重启。若此时未选择lightdm，而是gdm3，则输入如下命令，选中lightdm，并重启。如下图。

```terminal
sudo dpkg-reconfigure lightdm
```

![1](/assets/post/2021-11-11-SunloginDisconnectWin10/lightdm.png)

<br>

**下面的没用过。。。**

<https://www.cnblogs.com/xlpc/p/12345478.html>指出：

若需要卸载lightdm：

```terminal
sudo apt-get remove lightdm
```

如果安装了多个显示管理器，则可以使用以下方法在它们之间进行选择

```terminal
sudo dpkg-reconfigure gdm3
```

您可以在上面的命令中使用任何显示管理器的名称代替gdm3，它允许您在它们之间进行选择。您必须重新启动才能使更改生效。

要检查当前正在使用哪个显示管理器，请运行以下命令：

```terminal
cat /etc/X11/default-display-manager
```
