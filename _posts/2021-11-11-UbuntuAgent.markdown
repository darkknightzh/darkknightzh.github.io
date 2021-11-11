---
layout: post
title:  "Ubuntu使用代理"
date:   2021-11-11 14:30:00 +0800
tags: [linux]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>


转载请注明出处：

<https://darkknightzh.github.io/posts/SunloginDisconnectWin10>


参考网址：

<http://loonlog.com/2020/10/5/v2ray-server-new/>

<https://github.com/jiangxufeng/v2rayL>

<https://bella722.github.io/post/a2e7ced5.html>

<https://zhuanlan.zhihu.com/p/46973701>

<br>

说明：使用代理请用于学习，请勿用于发表和从事任何不利于国家安全、民族团结和国家复兴的言论和行为。

# P1 配置代理

1 配置代理服务器，安装v2ray等，确保不被河蟹，如<http://loonlog.com/2020/10/5/v2ray-server-new/>

2 使用<https://github.com/jiangxufeng/v2rayL>配置相关代理连接：

说明：不记得是用官方方法安装还是用下面网址安装的了。。。官方是ubuntu18.04的，我是用的系统是ubuntu 20.04的。下面是ubuntu 20.04的。

<https://bella722.github.io/post/a2e7ced5.html>

以下是官方安装方式：

① 安装

```terminal
bash <(curl -s -L http://dl.thinker.ink/install.sh)
```

② 更新

```terminal
bash <(curl -s -L http://dl.thinker.ink/update.sh)
```

③ 卸载

```terminal
bash <(curl -s -L http://dl.thinker.ink/uninstall.sh)
```
官方安装到此结束。下面的都一样了

④ 展示：点on打开代理服务器，点off关闭代理服务器

![1](/assets/post/2021-11-11-UbuntuAgent/1.png)
_图1_

⑤ 添加vmess等代理服务器：复制vmess的url后，粘贴到下面框中，需要点击回车，才会添加vmess服务器。也可以通过二维码添加。

![2](/assets/post/2021-11-11-UbuntuAgent/2.png)
_图2_

⑥ 需要注意的是，使用http或者socks协议时，要在具体软件中设置正确的端口号，和下面的一致。否则还是无法正常使用。
 
![3](/assets/post/2021-11-11-UbuntuAgent/3.png)
_图3_

⑦ V2rayL还有全局代理。默认是关闭全局代理。这时若打开图1中相关代理，想使用firefox或者终端能科学上网，还需要进行相应设置（见下面），若选择白名单模式，则若打开图1中相关代理后，firefox或者终端均可直接科学上网。算是比较省事的方法。但也有一个问题，如果代理打开，此处切换白名单模式和关闭全局代理，电脑会瞬间断网（使用向日葵时会连接已断开，需要重新连接）。如果关闭全局代理，则打开或关闭图1中代理时，不会出现这种问题。

说明：黑名单模式不清楚
 
![4](/assets/post/2021-11-11-UbuntuAgent/4.png)
_图4_

# P2 firefox设置

## P2.1 v2rayL关闭全局代理

firefox若需要使用代理，点击settings-General，滑到最下的Network Settings，点击Settings，默认应该是Use system proxy settings，可改为Manual proxy configuration，按图5设置，将HTTP Proxy和SOCKS Host均设置为127.0.0.1，端口分别为1081和1080（要和V2rayL中系统设置的端口号一致）。而后打开图1中代理开关，应该就可以使用firefox科学上网了。

需要注意的是：如果图1中代理关闭，使用图5设置，firefox无法上网。需要把下图中代理改成Use system proxy settings。这是比较麻烦的地方。

## P2.2 v2rayL白名单模式

打开图1中代理后，firefox无需任何设置，可直接科学上网。

关闭图1中代理后，firefox无需任何设置，可直接上墙内网。
 
![5](/assets/post/2021-11-11-UbuntuAgent/5.png)
_图5_

# P3 终端使用代理

终端上网，只pip等能安装软件。

## P3.1 v2rayL关闭全局代理

如<https://zhuanlan.zhihu.com/p/46973701>所说，把代理服务器地址写入.bashrc

```terminal
export http_proxy="http://127.0.0.1:1081"
export https_proxy="http://127.0.0.1:1081"
```

或者走socket5协议（ss,ssr）的话，代理端口是1080

```terminal
export http_proxy="socks5://127.0.0.1:1080"
export https_proxy="socks5://127.0.0.1:1080"
```

或者干脆直接设置ALL_PROXY

```terminal
export ALL_PROXY=socks5://127.0.0.1:1080
```

最后在执行如下命令应用设置

```terminal
source ~/.bashrc
```

因为对上面命令不太了解，我直接用的：

```terminal
export ALL_PROXY=socks5://127.0.0.1:1080
```

这样在图1中代理打开时，能科学上网，但是图1中代理关闭时，无法上网，需要如下命令：

```terminal
unset all_proxy
unset ALL_PROXY
```

也可以unset http_proxy和https_proxy：

```terminal
unset http_proxy
unset https_proxy
```

此时可以上墙内网站。

## P3.2 v2rayL白名单模式

打开图1中代理后，终端可直接科学上网，如curl www.google.com会返回结果。

关闭图1中代理后，终端只能访问墙内网站。curl www.google.com会卡住（之后没等着具体结果了）。

## P3.3 python使用socks5

可能会提示Missing dependencies for SOCKS support" when using SOCKS5 from Terminal，此时需要安装pysocks

① 若已将代理服务器地址写入.bashrc，则可unset相应变量后，使用如下命令来安装pysocks：

```terminal
pip install pysocks
```

② 若未将代理服务器地址写入.bashrc，除非程序中使用了pysocks，否则不会出现这个错误吧。。。此时可直接使用上述命令安装pysocks。

## P3.4 dpkg好像并不受代理的影响

这个不确定。。。
