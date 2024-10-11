from flask import Flask, request, Response
from werkzeug.utils import secure_filename
import os

from inference import get_prediction

app = Flask(__name__)

# 处理中文编码
app.config['JSON_AS_ASCII'] = False


# 跨域支持
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


app.after_request(after_request)

# 设置图片保存文件夹
app.config['UPLOAD_FOLDER'] = './static/images'

# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['jpg', 'jpeg']


# 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS


# 上传图片
@app.route("/uploadRecognizePlanet", methods=['POST', "GET"])
def uploadRecognize():
    if request.method == 'POST':
        # 获取post过来的文件名称，从name=file参数中获取
        file = request.files['file']
        # 检测文件格式
        if file and allowed_file(file.filename):
            # secure_filename方法会去掉文件名中的中文
            file_name = secure_filename(file.filename)
            # 保存图片
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            # 对上传的图片进行推理
            prediction_top3 = get_prediction(app.config['UPLOAD_FOLDER']+'/'+file_name)
            # 只返回最高置信度的数据 获取字典的第一个键值对
            prediction_top1 = {}
            first_key, first_value = next(iter(prediction_top3.items()))
            prediction_top1[first_key] = first_value
            print(prediction_top1)
            return {"data": prediction_top1, "message": "上传分析成功"}
        else:
            return "格式错误，仅支持jpg、jpeg格式文件"
    return {"code": '503', "data": "", "message": "仅支持post方法"}


@app.route("/recognize")
def recognize(file_name):
    prediction = get_prediction(app.config['UPLOAD_FOLDER'] + '/' + file_name)

    return {"data": prediction, "message": "分析成功"}

# 查看图片
@app.route("/images/<imageId>")
def get_frame(imageId):
    # 图片上传保存的路径
    with open(r'./static/images/{}'.format(imageId), 'rb') as f:
        image = f.read()
        result = Response(image, mimetype="image/jpg")
        return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
