---
layout: post
title:  "ubuntu中将文件夹打包成iso的命令"
date:   2018-03-14 15:20:00 +0800
tags: [linux]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/CreateIsoInUbuntu>

原始cnblogs网址：

<http://www.cnblogs.com/darkknightzh/p/8564483.html>

参考网址：

<https://zhidao.baidu.com/question/2203263841064787548.html>

<https://zhidao.baidu.com/question/680880446035239332.html>

<http://man.linuxde.net/mkisofs>

<br>

要将某个文件夹打包成iso，减少硬盘中的文件数量，可以在需要打包的文件夹的父文件夹中使用下面的命令。

```terminal
$ mkisofs -o aa.iso -J -R -V bb ccFloder
```

其中 aa.iso为需要打包成的iso文件名，-V后面的bb为指定光盘的卷册集ID，ccFloder为需要打包的文件夹名字。

如果需要保持原始文件名，要添加-J参数，否则打包后，文件名全改变了。见第一个参考网址。

如果需要排除部分文件夹，可以使用-x excludefolder，具体见第二个网址。

所有的参数说明，见第三个网址。

<span style="color:#ff9900; font-weight:bold"> =================================================== </span>

**211101更新：**

之前在ubuntu系统中就是通过上述命令打包iso文件，但是在windows10的ubuntu 20子系统中，使用上述命令后，（猜测）会对文件进行编码，如下图所示。之前只会对图片使用后三位编码，不清楚具体为什么，现在再试，无法复现。。。

![1](/assets/post/2018-03-14-CreateIsoInUbuntu/1.png)

反正如果文件比较多时，假如提示无法对某文件进行编码，则可以使用如下命令（加上'-l'，此处为小写的'l'），不对文件名重新编码（实际iso中存储的还是原来文件名）。

```terminal
$ mkisofs -l -o aa.iso -J -R -V bb -input-charset iso8859-1 ccFloder
```

下面为不加-input-charset iso8859-1的提示：

```text
I: -input-charset not specified, using utf-8 (detected in locale settings)
```

**211101更新结束**

<span style="color:#ff9900; font-weight:bold"> =================================================== </span>
