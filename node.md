# 版本号
v4.1.1

# 使用方法

1、setting中将sound的输出更换为`Analog Output-USB Audio Device`

2、通过`python3 main.py`启动程序


## 版本介绍
1、检测的延时显示

2、支撑yolo的pt和pth格式模型更换

3、无需输入串口权限指令

4、自动保留两个检测结果，节省result文件夹的空间

## 目录结构
    audio       -- 音频
    landmarks   -- 关键点检测器
    models      -- yolo的模型
    result      -- 存放运行结果
    UI          -- qt的布局和图标
    utils       -- yolo需要用的工具类
    weights     -- 权重
        IMU.py      -- 陀螺仪
        Driver.py   --驾驶员类
        main.py     -- QT界面主函
        export.py   -- 接口
        Sender.py   -- 短信发送

## 功能实现
1、声音提醒完成。优化了声音播放的机制，并区分不同状态下的声音播放
    
2、车辆状态完成

3、阈值调整完成

4、短信提醒

5、检测的延时显示

6、支撑yolo的pt和pth格式模型更换

7、修改串口访问权限问题，无需输入指令（灯带的机器）

## 问题

#### 声音播放重载需要一定的时间间隔

##### 思路

这里可以通过qt内部事件进行中断。

#### IMU的串口权限问题，每一次开机都需要配置一次`sudo chmod 777 /dev/ttyUSB0`
已解决
##### 方法如下
将用户添加到dialout组，从而使其能够访问串口设备

通过命令`sudo usermod -aG dialout <username>`，实现上述内容。请将<username>替换为您要添加到dialout组的实际用户名。

重新启动计算机或注销当前用户并重新登录，以使更改生效。

## 实现流程
main.py
    
    该文件是程序的主入口。

    实例化界面（绑定个按钮事件、定时器初始化、驾驶员实例、音频播放实例）

    show_video_frame函数是检测的主体逻辑逻辑
    
IMU.py      

    陀螺仪的数据来源
    
    通过DueData()函数获取角度的信息，返回的数据格式是一个保留的小数的角度信息列表`[angle_x, angle_y, angle_z]`
  
  
Driver.py   
        
    驾驶类别
    
    包含驾驶员的各项数值记录、车辆状态监控。
    
    数值记录：简单的数值更新与驾驶员状态监控

    车辆状态监控：通过QT的子线程，实时读取IMU的角度，并比较角度的变化得到车辆状态

export.py
    
    检测算法提供的一些接口函数

Sender.py

    阿里巴巴的短信报文发送

    本质是一个http的请求。可以拿到报文，根据resq进行后续的信息监控

