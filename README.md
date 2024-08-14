# KuiperInfer (自制深度学习推理框架)
**我们在本项目的基础上开设了一个全新的课程，支持CUDA、Int8量化推理，《从零自制大模型推理框架》，以下是目录，感兴趣的同学可以扫描二维码了解，欢迎大家参加。**

<img src="imgs/me.png" style="zoom: 33%;" />

## 《动手自制大模型推理框架》常见问题

1. **课程更新完了吗？**

主体已经更新完毕，完全可以开始自习。支持cuda, int8推理。

2. **这是收费课程吗？怎么收费，怎么付款，过段时间买可以吗？**

收费课程，256，微信转账。可以，但是我微信好友快满了，隔一段时间会清理，而且不定期会涨价。

3. **为什么b站会贵一点，内容都一样吗？**

b站要抽成，内容都一样

4. **怎么看课**

飞书网盘，打开浏览器就可以看

5. **有答疑吗**

有的，且有答疑群，群友也很热情。

6. **不会cpp可以学吗？**

事在人为，我也尽量深入浅出教学

7. **课程目录有吗**

见下文

9. **作者是干嘛的？**

主业就是开发大模型推理框架的，课件已经被人民邮电出版社约稿，同时也是kuiperinfer项目，也就是本项目的发起人，目前全github cpp项目排名120位。

**《动手自制大模型推理框架》项目运行效果fp32模型1.1b参数。**
![](./imgs/do.gif)

KuipeInfer目前2.3k star，帮助很多人获得了大厂岗位。
## 《动手自制大模型推理框架》课程目录

**一、项目整体架构和设计**

> 学习架构思维，防止自己只会优化局部实现

1. 环境的安装和课程简介
2. 资源管理和内存管理类的设计与实现
3. 张量类的设计与实现
4. 算子类的设计与实现
5. 算子的注册和管理

**二、支持LLama2模型结构**
> 本节将为大家补齐算法工程师思维，在算法层面讲解大模型和Transformer的原理之后，开始对LLama2进行支持


6. LLama模型的分析
7. MMap内存映射技术打开大模型的权重文件
8. LLama模型文件的参数和权重载入
9. LLama中各个层的初始化以及输入张量、权重张量的分配和申请
10. 实现大模型中的KV Cache机制

**三、模型的量化**

> 为了减少显存的占用，我们开发了int8模型量化模块

11. 量化模型权重的导出
12. 量化系数和权重的加载
13. 量化乘法算子的实现

**四、Cuda基础和算子实现**

> 带你学Cuda并在实战大模型算子的实现，为大模型推理赋能

14. Cuda基础入门1 - 内容待定
15. Cuda基础入门2 - 内容待定
16. Cuda基础入门3 - 内容待定
17. Cuda基础入门4 - 内容待定
18. RMSNorm算子的Cuda实现
19. Softmax算子的Cuda实现
20. Add算子的Cuda实现
21. Swiglu算子的Cuda实现
22. GEMV算子的Cuda实现
23. 多头注意力机制的Cuda实现
24. 让框架增加Cuda设备的支持和管理
25. 完成Cuda推理流程

**五、用推理框架做点有趣的事情**

26. 文本生成
27. 讲一段小故事
28. 让大模型和你进行多轮对话


**六、学习其他商用推理框架的实现，查漏补缺**

29. LLama.cpp的设计和实现讲解
30. Miopen（AMD出品，对标CUDNN）的设计和实现讲解
31. 总结
