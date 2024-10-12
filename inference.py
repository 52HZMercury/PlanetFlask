import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd

def get_prediction(planet_img):
    # 载入 onnx 模型，获取 ONNX Runtime 推理器
    ort_session = onnxruntime.InferenceSession('model/resnet18_planet.onnx')

    # test
    # # 构造随机输入，获取输出结果
    # x = torch.randn(1, 3, 256, 256).numpy()
    #
    # # onnx runtime 输入
    # ort_inputs = {'input': x}
    # # onnx runtime 输出
    # ort_output = ort_session.run(['output'], ort_inputs)[0]
    # # 注意，输入输出张量的名称需要和 torch.onnx.export 中设置的输入输出名对应
    #

    # 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Grayscale(num_output_channels=3),  # 确保三通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_path = planet_img
    img_pil = Image.open(img_path)

    input_img = test_transform(img_pil)
    input_tensor = input_img.unsqueeze(0).numpy()

    # ONNX Runtime 输入
    ort_inputs = {'input': input_tensor}

    # ONNX Runtime 输出
    pred_logits = ort_session.run(['output'], ort_inputs)[0]
    pred_logits = torch.tensor(pred_logits)
    # 对 logit 分数做 softmax 运算，得到置信度概率
    pred_softmax = F.softmax(pred_logits, dim=1)

    # 取置信度最高的前 n 个结果
    n = 3
    top_n = torch.topk(pred_softmax, n)

    # 预测类别
    pred_ids = top_n.indices.numpy()[0]
    # 预测置信度
    confs = top_n.values.numpy()[0]

    # 载入类别 ID 和 类别名称 对应关系
    idx_to_labels = np.load('mapping/idx_to_labels.npy', allow_pickle=True).item()
    # print(idx_to_labels)

    # idx_to_labels = {}
    # for idx, row in df.iterrows():
    #     idx_to_labels[row['ID']] = row['class']   # 英文
    # #     idx_to_labels[row['ID']] = row['Chinese'] # 中文

    results = {}
    # 分别用英文和中文打印预测结果
    for i in range(n):
        class_name = idx_to_labels[pred_ids[i]] # 获取类别名称
        confidence = confs[i] * 100             # 获取置信度
        results[class_name] = confidence

    return results
