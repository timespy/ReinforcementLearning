# ReinforcementLearning
the learning code for reinforcement learning

刚刚学完Qlearning的相关知识，做了一个简单的demo来测试一下学习效果。在这个迷宫中，大约学习90次左右的时候，探索者才能顺利的找到宝藏。不过这和

迷宫的设计应该也有关，因为我在设置撞墙的时候（不是进入陷阱），状态不变，且返回奖励同走到普通道路上。进入陷阱，奖励为-1，得到宝藏奖励为1，撞墙如果

改成奖励为-0.5应该会更好一些。
