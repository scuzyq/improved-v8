from ultralytics import YOLO

# 安装命令
# python setup.py develop

# 数据集示例百度云链接
# 链接：https://pan.baidu.com/s/19FM7XnKEFC83vpiRdtNA8A?pwd=n93i 
# 提取码：n93i 

if __name__ == '__main__':
    # # 直接使用预训练模型创建模型.
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/exp1.yaml', 'data':'dataset/data.yaml'})
    
    # # 使用yaml配置文件来创建模型,并导入预训练权重.
    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model.load('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/exp1.yaml', 'data':'dataset/data.yaml'})
    
    # # 模型验证
    # model = YOLO('runs/detect/yolov8n_exp/weights/best.pt')
    # model.val(**{'data':'dataset/data.yaml'})
    
    # # 模型推理
    # model = YOLO('runs/detect/train53/weights/best.pt')
    # model.predict(source='datasets/test/images', **{'save':True})

    model = YOLO('ultralytics/cfg/models/v8/yolov8-att.yaml')
    model.load('yolov8n.pt')
    # model = YOLO('yolov8n.pt')
    model.train(**{'cfg':'ultralytics/cfg/exp1.yaml', 'data':'datasets/data.yaml'})

    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model.train(data='datasets/data.yaml')