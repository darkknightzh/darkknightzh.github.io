---
layout: post
title:  "windows10和ubuntu双系统的时间差"
date:   2021-11-11 14:00:00 +0800
tags: [linux]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>


原始网址：
<https://www.cnblogs.com/chengjue924/p/8915758.html>

```terminal
sudo apt-get install ntpdate
sudo ntpdate time.windows.com
sudo hwclock --localtime --systohc
```
