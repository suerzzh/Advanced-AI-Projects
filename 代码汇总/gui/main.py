import argparse
import random
import sys
import torch
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QWidget, QDialog,\
                              QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox,\
                              QHBoxLayout
from PySide6 import QtGui, QtCore
from PySide6.QtGui import QPixmap, QImage, QFont, QFontDatabase, QBrush, QPalette, QTransform
from PySide6.QtCore import QTimer, Qt, Signal
import re
from utils.main_window import Ui_MainWindow
from threading import Thread
from utils.plot_mask import draw_detections

import argparse
import os
import sys
import shutil
from pathlib import Path
from utils.database import SqliteDBOperator
import warnings
warnings.filterwarnings("ignore")

import cv2
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOTS = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils import camera_rc

def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



class LoginWidget(QWidget):
    login_success_signal = Signal()

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setFixedSize(900, 500)
        self.setWindowTitle("基于深度学习的船舰检测系统-登录")

    def init_ui(self):
        # 创建系统名称标签的字体对象
        system_name_font = QFont("华文楷体")
        system_name_font.setPointSize(24)

        # 创建用户名、密码的字体对象
        custom_font = QFont("华文楷体")
        custom_font.setPointSize(11)

        layout = QVBoxLayout()

        # 添加显示系统名称的标签
        system_name_label = QLabel('基于深度学习的船舰检测系统')
        system_name_label.setFont(system_name_font)
        system_name_label.setStyleSheet("color: white;")
        system_name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(system_name_label)

        # 创建放置输入框和按钮的部件
        central_widget = QWidget()
        central_layout = QVBoxLayout()

        self.username_label = QLabel('用户名:')
        self.username_edit = QLineEdit()
        self.username_edit.setFixedWidth(200)
        self.password_label = QLabel('密码:')
        self.password_edit = QLineEdit()
        self.password_edit.setFixedWidth(200)
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.login_button = QPushButton('登录')
        self.login_button.setFixedWidth(93)
        self.login_button.clicked.connect(self.check_login)
        self.register_button = QPushButton('注册')
        self.register_button.setFixedWidth(93)
        self.register_button.clicked.connect(self.show_register_dialog)

        # 设置用户名、密码标签文字颜色
        self.username_label.setStyleSheet("color: white;")
        self.password_label.setStyleSheet("color: white;")

        self.username_label.setFont(custom_font)
        self.password_label.setFont(custom_font)
        self.login_button.setFont(custom_font)
        self.register_button.setFont(custom_font)

        # 调整布局，通过添加空白的QLabel来实现间距效果
        central_layout.addWidget(self.username_label)
        central_layout.addWidget(self.username_edit)
        central_layout.addWidget(self.password_label)
        central_layout.addWidget(self.password_edit)

        spacer_label_1 = QLabel()  # 用于在按钮上方增加一些空白间距
        spacer_label_1.setFixedHeight(150)  # 设置空白标签
        spacer_label_2 = QLabel()
        spacer_label_2.setFixedHeight(5)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.register_button)
        central_layout.addWidget(spacer_label_2)
        central_layout.addLayout(button_layout)
        central_layout.addWidget(spacer_label_1)
        central_widget.setLayout(central_layout)

        layout.addWidget(central_widget, alignment=Qt.AlignCenter)
        self.setLayout(layout)

        # 设置背景图片
        self.setAutoFillBackground(True)
        self.update_background_image()  # 初始化时先设置一次背景图片

    def update_background_image(self):
        palette = self.palette()
        pixmap = QPixmap("./ui_images/login.png")
        pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                              Qt.TransformationMode.SmoothTransformation)
        rect = self.rect()
        brush = QBrush(pixmap)
        brush.setTransform(QTransform().translate((rect.width() - pixmap.width()) / 2,
                                                 (rect.height() - pixmap.height()) / 2))
        palette.setBrush(QPalette.ColorRole.Window, brush)
        self.setPalette(palette)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_background_image()  # 窗口尺寸改变时，重新更新背景图片的绘制
        
    def check_login(self):
        username = self.username_edit.text()
        password = self.password_edit.text()
        db_operator = SqliteDBOperator()
        if db_operator.check_user_login(username, password):
            self.login_success_signal.emit()  # 发射信号
            self.close()
            db_operator.close_connection()
        else:
            QMessageBox.warning(self, "登录失败", "账号或密码错误，请重新输入", QMessageBox.StandardButton.Ok)
            db_operator.close_connection()

    def show_register_dialog(self):
        
        register_dialog = QDialog(self) 
        register_dialog.setWindowTitle('注册')
        register_layout = QVBoxLayout()

        register_username_label = QLabel('用户名:')
        register_username_edit = QLineEdit()
        register_password_label = QLabel('密码:')
        register_password_edit = QLineEdit()
        register_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        register_confirm_password_label = QLabel('确认密码:')
        register_confirm_password_edit = QLineEdit()
        register_confirm_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        register_button = QPushButton('注册')
        register_button.clicked.connect(lambda: self.check_register_info(register_username_edit.text(),
                                                                         register_password_edit.text(),
                                                                         register_confirm_password_edit.text(),
                                                                         register_dialog))

        register_layout.addWidget(register_username_label)
        register_layout.addWidget(register_username_edit)
        register_layout.addWidget(register_password_label)
        register_layout.addWidget(register_password_edit)
        register_layout.addWidget(register_confirm_password_label)
        register_layout.addWidget(register_confirm_password_edit)
        register_layout.addWidget(register_button)

        register_dialog.setLayout(register_layout)
        register_dialog.exec()

    def check_register_info(self, username, password, confirm_password, register_dialog):
        # 检查用户名是否为空
        if username == "":
            QMessageBox.warning(self, "注册失败", "用户名为空", QMessageBox.StandardButton.Ok)
            return
        # 检查用户名长度是否符合要求
        elif len(username) < 4:
            QMessageBox.warning(self, "注册失败", "用户名长度不符合要求，至少4位", QMessageBox.StandardButton.Ok)
            return
        elif len(username) > 12:
            QMessageBox.warning(self, "注册失败", "用户名长度不符合要求，至多12位", QMessageBox.StandardButton.Ok)
            return
        # 新增：检查用户名格式是否符合要求
        elif not re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', username):
            QMessageBox.warning(self, "注册失败", "用户名格式不符合要求，只能包含大小写字母和数字，且不能以数字开头", QMessageBox.StandardButton.Ok)
            return
        # 检查两次密码输入是否一致
        elif password!= confirm_password:
            QMessageBox.warning(self, "注册失败", "两次密码输入不一致", QMessageBox.StandardButton.Ok)
            return
        # 检查密码长度（至少6位，至多20位）
        elif len(password) < 6:
            QMessageBox.warning(self, "注册失败", "密码长度不符合要求，至少6位", QMessageBox.StandardButton.Ok)
            return
        elif len(password) > 20:
            QMessageBox.warning(self, "注册失败", "密码长度不符合要求，至多20位", QMessageBox.StandardButton.Ok)
            return
        # 检查密码是否至少包含数字、字母、特殊符号中的两种
        categories = [r'\d', r'[a-zA-Z]', r'[^\w]'] 
        count = 0
        for category in categories:
            if re.search(category, password):
                count += 1
        if count < 2:
            QMessageBox.warning(self, "注册失败", "密码至少需包含数字、字母、特殊符号中的两种", QMessageBox.StandardButton.Ok)
            return
        else:
            db_operator = SqliteDBOperator()
            # 检查用户名是否已存在
            if db_operator.check_user_exists(username):
                QMessageBox.warning(self, "注册失败", "该用户已存在，请直接登录", QMessageBox.StandardButton.Ok)
                return
            db_operator.insert_user_data(username, password)
            msg_box = QMessageBox.information(self, "注册成功", "注册成功", QMessageBox.StandardButton.Ok)
            # 注册成功后，关闭注册对话框，回到登录界面
            if msg_box == QMessageBox.StandardButton.Ok:
                register_dialog.close()


            
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.input_width = self.input.width()
        self.input_height = self.input.height()
        # self.output_width = self.output.width()
        # self.output_height = self.output.height()
        self.imgsz = 640
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer_c = QTimer(self)
        self.timer_c.timeout.connect(self.detect_camera)
        self.video = None
        self.out = None
        # 若是cpu检测，将cuda:0替换成cpu
        self.device = "cuda:0"
        self.num_stop = 1
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.results = []
        self.camera = None
        self.running = False
        self.bind_slots()
        self.init_icons()

    def open_image(self):
        self.timer.stop()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "./", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        if self.file_path:
            dialog = QFileDialog(self, "Open File", self.file_path)
            dialog.resize(800, 600)
            dialog.close()
            pixmap = QPixmap(self.file_path)
            scaled_pixmap = pixmap.scaled(640, 480, aspectMode=Qt.KeepAspectRatio)
            # scaled_pixmap = pixmap.scaledToWidth(self.input.width())
            # scaled_pixmap = pixmap.scaled(self.input.size(), aspectMode=Qt.KeepAspectRatio)
            self.input.setPixmap(QPixmap(self.file_path))
            self.lineEdit.setText('图片打开成功！！！')

    def open_video(self):
        self.timer.stop()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select vidos", dir='./', filter="Videos (*.mp4 *.avi *.gif *.MPEG)", options=options)
        if self.video_path:
            dialog = QFileDialog(self, "Open File", self.video_path)
            dialog.resize(800, 600)
            dialog.close()
            self.video_path = self.video_path
            self.video = cv2.VideoCapture(self.video_path)

            # 读取一帧用于展示
            ret, frame = self.video.read()
            if ret:
                self.lineEdit.setText("成功打开视频！！！")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dst_size = (self.input_width, self.input_height)
                resized_frame = cv2.resize(frame, dst_size, interpolation=cv2.INTER_AREA)
                self.input.setPixmap(QPixmap(convert2QImage(resized_frame)))
            else:
                self.lineEdit.setText("视频有误，请重新打开！！！")
            self.out = cv2.VideoWriter('prediction.mp4', cv2.VideoWriter_fourcc(
                    *'mp4v'), 30, (int(self.video.get(3)), int(self.video.get(4))))

    def load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.button_weight_select, '选择权重文件',
                                                                  'weights/', "Weights (*.pt *.onnx *.engine)", options=options)
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"权重打开失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            dialog = QFileDialog(self, "Open File", self.openfile_name_model)
            dialog.resize(800, 600)
            dialog.close()
            result_str = '成功加载模型权重, 权重地址: ' + str(self.openfile_name_model)
            self.lineEdit.setText(result_str)

    def init_model(self):
        from ultralytics import YOLO

        self.weights_path = str(self.openfile_name_model)
        # 8.0.124
        self.model = YOLO(self.weights_path)
        self.names = self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        print("model initial done")

        QtWidgets.QMessageBox.information(self, u"!", u"模型初始化成功", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        self.lineEdit.setText("成功初始化模型!!!")

    def detect_begin(self):
        # name_list = []
        self.img = cv2.imread(self.file_path)
        self.pred = self.model.predict(source=self.img, iou=self.numiou, conf=self.numcon)  # save plotted images
        preprocess_speed = self.pred[0].speed['preprocess']
        inference_speed = self.pred[0].speed['inference']
        postprocess_speed = self.pred[0].speed['postprocess']
        self.lineEdit_detect_time.setText(str(round((preprocess_speed + inference_speed + postprocess_speed) / 1000, 3)))
        self.lineEdit_detect_object_nums.setText(str(self.pred[0].boxes.conf.shape[0]))

        self.results = self.pred[0].boxes.xyxy.tolist()

        if self.pred[0].boxes.conf.shape[0]:
            for i in range(self.pred[0].boxes.conf.shape[0]):
                self.comboBox.addItem('目标' + str(i + 1))
        
        QtWidgets.QMessageBox.information(self, u"!", u"成功检测图像", buttons=QtWidgets.QMessageBox.Ok,
                                      defaultButton=QtWidgets.QMessageBox.Ok)
    
        self.lineEdit.setText("成功检测图像!") 

    def convert2QImage(img):
        height, width, channel = img.shape
        return QImage(img, width, height, width * channel, QImage.Format_RGB888)

    def detect_show(self):
        conf_list = self.pred[0].boxes.conf.tolist()
        cls_list_int = [int(i) for i in self.pred[0].boxes.cls.tolist()]
        xyxy_list_int = [[round(num) for num in sublist] for sublist in self.pred[0].boxes.xyxy.tolist()]

        self.combined_image = draw_detections(self.img, xyxy_list_int, conf_list, cls_list_int, 0.4)

        self.result = cv2.cvtColor(self.combined_image, cv2.COLOR_BGR2BGRA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        
        self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.input.setScaledContents(True)  # 自适应界面大小

        self.lineEdit.setText('图片检测成功！')

        cv2.imwrite(f'prediction.jpg', self.combined_image)
        
    # 视频检测
    def detect_video(self):
        self.timer.start()
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
            self.video.release()
            self.out.release()
        else:
            name_list = []   

            self.comboBox.clear()

            # 检测每一帧
            self.pred = self.model.predict(source=frame, iou=self.numiou, conf=self.numcon)  # save plotted images
            preprocess_speed = self.pred[0].speed['preprocess']
            inference_speed = self.pred[0].speed['inference']
            postprocess_speed = self.pred[0].speed['postprocess']
            self.lineEdit_detect_time.setText(str(round((preprocess_speed + inference_speed + postprocess_speed) / 1000, 2)))
            self.lineEdit_detect_object_nums.setText(str(self.pred[0].boxes.conf.shape[0]))

            self.results = self.pred[0].boxes.xyxy.tolist()

            if self.pred[0].boxes.conf.shape[0]:
                for i in range(self.pred[0].boxes.conf.shape[0]):
                    self.comboBox.addItem('目标' + str(i + 1))

            # 画图
            conf_list = self.pred[0].boxes.conf.tolist()
            cls_list_int = [int(i) for i in self.pred[0].boxes.cls.tolist()]
            xyxy_list_int = [[round(num) for num in sublist] for sublist in self.pred[0].boxes.xyxy.tolist()]

            self.combined_image = draw_detections(frame, xyxy_list_int, conf_list, cls_list_int, 0.4)

            # 写视频
            self.out.write(self.combined_image)

            self.result_frame = cv2.cvtColor(self.combined_image, cv2.COLOR_BGR2BGRA)
            self.QtImg = QtGui.QImage(
                self.result_frame.data, self.result_frame.shape[1], self.result_frame.shape[0], QtGui.QImage.Format_RGB32)
            
            self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.input.setScaledContents(True)  # 自适应界面大小
            self.lineEdit.setText('正在检测视频！！！')

    def suspend_video(self):
        self.timer.blockSignals(False)
        if self.timer.isActive() == True and self.num_stop % 2 == 1:
            self.button_video_suspend.setText(u'继续视频检测')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer.blockSignals(True)
        else:
            self.num_stop = self.num_stop + 1
            self.button_video_suspend.setText(u'暂停视频检测')

    def stop_video(self):
        if self.num_stop % 2 == 0:
            self.video.release()
            self.out.release()
            self.input.setPixmap(QPixmap())
            self.input.setScaledContents(True)
            # self.output.setPixmap(QPixmap("input.png"))
            # self.output.setScaledContents(True)
            self.button_video_suspend.setText(u'暂停视频检测')
            self.num_stop = self.num_stop + 1
            self.timer.blockSignals(False)
            self.lineEdit_detect_time.clear()
            self.lineEdit_detect_object_nums.clear()
            self.lineEdit_xmin.clear()
            self.lineEdit_ymin.clear()
            self.lineEdit_xmax.clear()
            self.lineEdit_ymax.clear()
            self.lineEdit.clear()
        else:
            self.video.release()
            self.out.release()
            self.input.setPixmap(QPixmap())
            self.input.setScaledContents(True)
            # self.output.clear()
            self.timer.blockSignals(False)
            self.lineEdit_detect_time.clear()
            self.lineEdit_detect_object_nums.clear()
            self.lineEdit_xmin.clear()
            self.lineEdit_ymin.clear()
            self.lineEdit_xmax.clear()
            self.lineEdit_ymax.clear()
            self.lineEdit.clear()
            
    def stop_image(self):
        self.input.setPixmap(QPixmap())
        self.input.setScaledContents(True)
        self.lineEdit_detect_time.clear()
        self.lineEdit_detect_object_nums.clear()
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()
        self.comboBox.clear()
        self.lineEdit.clear()

    def export_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #self.OutputDir, _ = QFileDialog.getSaveFileName(
        #    self,  # 父窗口对象
        #    "导出图片",  # 标题
        #    r".",  # 起始目录
        #    "图片类型 (*.jpg *.jpeg *.png *.bmp)",  # 选择类型过滤项，过滤内容在括号中
        #    options=options
        #)

        #if self.OutputDir == "":
        #    QtWidgets.QMessageBox.warning(self, '提示', '请先选择图片保存的位置')
        #else:
        try:
            #dialog = QFileDialog(self, "Save image", self.OutputDir)
            #dialog.resize(800, 600)
            #dialog.close()
            cv2.imwrite("./save_data/pred.jpg", self.combined_image)
            QtWidgets.QMessageBox.warning(self, '提示', '导出成功!')
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, '提示', '请先完成识别工作')
            print(e)

    def export_videos(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        #self.OutputDirs, _ = QFileDialog.getSaveFileName(
        #    self,  # 父窗口对象
        #    "导出视频",  # 标题
        #    r".",  # 起始目录
        #    "图片类型 (*.mp3 *.mp4 *.gif *.avi)",  # 选择类型过滤项，过滤内容在括号中
        #    options=options
        #)
        #if self.OutputDirs == "":
        #    QtWidgets.QMessageBox.warning(self, '提示', '请先选择视频保存的位置')
        #else:
        self.out.release()
        try:
            #dialog = QFileDialog(self, "Save video", self.OutputDirs)
            #dialog.resize(800, 600)
            #dialog.close()
            shutil.copy(str(ROOT) + '/prediction.mp4', "./save_data/pred.mp4")
            QtWidgets.QMessageBox.warning(self, '提示', '导出成功!')
        except Exception as e:
            print(e)
            QtWidgets.QMessageBox.warning(self, '提示', '请先完成识别工作')
            
    def ValueChange(self):
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.con_number.setValue(self.numcon)
        self.iou_number.setValue(self.numiou)
    
    def Value_change(self):
        num_conf = self.con_number.value()
        num_ious = self.iou_number.value()
        self.con_slider.setValue(int(num_conf * 100))
        self.iou_slider.setValue(int(num_ious * 100))
        self.numcon = num_conf
        self.numiou = num_ious
        
    def value_change_comboBox(self):
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()
        object = self.comboBox.currentText()
        if object:
            object_number_str = object[-1]
            object_number_int = int(object_number_str)
            object_number_index = object_number_int - 1
            if self.results:
                self.lineEdit_xmin.setText(str(int(self.results[object_number_index][0])))
                self.lineEdit_ymin.setText(str(int(self.results[object_number_index][1])))
                self.lineEdit_xmax.setText(str(int(self.results[object_number_index][2])))
                self.lineEdit_ymax.setText(str(int(self.results[object_number_index][3])))

    def open_camera(self):
        self.lineEdit.setText("打开摄像头中...")
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.lineEdit.setText("成功打开摄像头！")
            self.timer_c.start(30)

    def detect_camera(self):
        ret, frame = self.camera.read()
        if ret:
            result_input = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            self.QtImg_input = QtGui.QImage(
                result_input.data, result_input.shape[1], result_input.shape[0], QtGui.QImage.Format_RGB32)
            self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg_input))
            self.input.setScaledContents(True)

            if self.running:
                name_list = []
                self.comboBox.clear()
                # frame_showing = frame
                # 检测每一帧
                self.pred = self.model.predict(source=frame, conf=self.numcon, iou=self.numiou)  # save plotted images
                preprocess_speed = self.pred[0].speed['preprocess']
                inference_speed = self.pred[0].speed['inference']
                postprocess_speed = self.pred[0].speed['postprocess']
                self.lineEdit_detect_time.setText(str(round((preprocess_speed + inference_speed + postprocess_speed) / 1000, 2)))
                self.lineEdit_detect_object_nums.setText(str(self.pred[0].boxes.conf.shape[0]))

                self.results = self.pred[0].boxes.xyxy.tolist()

                if self.pred[0].boxes.conf.shape[0]:
                    for i in range(self.pred[0].boxes.conf.shape[0]):
                        self.comboBox.addItem('目标' + str(i + 1))

                # 画图
                conf_list = self.pred[0].boxes.conf.tolist()
                cls_list_int = [int(i) for i in self.pred[0].boxes.cls.tolist()]
                xyxy_list_int = [[round(num) for num in sublist] for sublist in self.pred[0].boxes.xyxy.tolist()]

                self.combined_image = draw_detections(frame, xyxy_list_int, conf_list, cls_list_int, 0.4)

                self.result_frame = cv2.cvtColor(self.combined_image, cv2.COLOR_BGR2BGRA)
                self.QtImg = QtGui.QImage(
                    self.result_frame.data, self.result_frame.shape[1], self.result_frame.shape[0], QtGui.QImage.Format_RGB32)
                
                self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.input.setScaledContents(True)  # 自适应界面大小

                self.lineEdit.setText('正在使用摄像头进行检测！！！')
                
        else:
            self.timer_c.stop()
            self.camera.release()
            self.camera = None

    def close_camera(self):
        self.running = False
        self.camera = None
        self.timer_c.stop()
        self.input.setPixmap(QPixmap())
        self.input.setScaledContents(True)
        # self.QtImg = None
        # self.output.setPixmap(QtGui.QPixmap())
        self.lineEdit.setText("已关闭摄像头！")
        self.lineEdit_detect_time.clear()
        self.lineEdit_detect_object_nums.clear()
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()

    def detect_camera_running(self):
        self.running = True

    def bind_slots(self):
        self.buttton_image_select.clicked.connect(self.open_image)
        self.buttton_video_select.clicked.connect(self.open_video)
        self.button_weight_select.clicked.connect(self.load_model)
        self.button_weight_init.clicked.connect(self.init_model)
        self.button_image_detect.clicked.connect(self.detect_begin)
        self.button_image_show.clicked.connect(self.detect_show)
        self.button_video_detect.clicked.connect(self.detect_video)
        self.button_video_suspend.clicked.connect(self.suspend_video)
        self.button_video_stop.clicked.connect(self.stop_video)
        self.button_image_stop.clicked.connect(self.stop_image)
        self.button_image_export.clicked.connect(self.export_images)
        self.button_video_export.clicked.connect(self.export_videos)
        self.con_slider.valueChanged.connect(self.ValueChange)
        self.iou_slider.valueChanged.connect(self.ValueChange)
        self.con_number.valueChanged.connect(self.Value_change)
        self.iou_number.valueChanged.connect(self.Value_change)
        self.comboBox.currentTextChanged.connect(self.value_change_comboBox)
        self.timer.timeout.connect(self.detect_video)
        self.button_camera_start.clicked.connect(self.open_camera)
        self.button_camera_stop.clicked.connect(self.close_camera)
        self.button_camera_detect.clicked.connect(self.detect_camera_running)

    def init_icons(self):
        self.label_weight_select.setPixmap(QPixmap('ui_images/icons/weight.png'))
        self.label_weight_select.setScaledContents(True)
        self.label_weight_init.setPixmap(QPixmap('ui_images/icons/init.png'))
        self.label_weight_init.setScaledContents(True)
        self.label_image_select.setPixmap(QPixmap(':/image/icons/image.png'))
        self.label_image_select.setScaledContents(True)
        self.label_video_select.setPixmap(QPixmap(':/image/icons/video.png'))
        self.label_video_select.setScaledContents(True)
        self.label_image_detect.setPixmap(QPixmap(':/image/icons/recognition.png'))
        self.label_image_detect.setScaledContents(True)
        self.label_video_detect.setPixmap(QPixmap(':/image/icons/detect_video.png'))
        self.label_video_detect.setScaledContents(True)
        self.label_image_show.setPixmap(QPixmap(':/image/icons/image_result.png'))
        self.label_image_show.setScaledContents(True)
        self.label_video_suspend.setPixmap(QPixmap(':/image/icons/suspend_video.png'))
        self.label_video_suspend.setScaledContents(True)
        self.label_image_stop.setPixmap(QPixmap(':/image/icons/stop_image.png'))
        self.label_image_stop.setScaledContents(True)
        self.label_video_stop.setPixmap(QPixmap(':/image/icons/stop_video.png'))
        self.label_video_stop.setScaledContents(True)
        self.label_image_export.setPixmap(QPixmap(':/image/icons/export.png'))
        self.label_image_export.setScaledContents(True)
        self.label_video_export.setPixmap(QPixmap(':/image/icons/export.png'))
        self.label_video_export.setScaledContents(True)
        self.label_detect_time.setPixmap(QPixmap(':/image/icons/used_time.png'))
        self.label_detect_time.setScaledContents(True)
        self.label_detect_object_nums.setPixmap(QPixmap(':/image/icons/object_nums.png'))
        self.label_detect_object_nums.setScaledContents(True)
        self.label_detect_object_pos.setPixmap(QPixmap(':/image/icons/position.png'))
        self.label_detect_object_pos.setScaledContents(True)
        self.label_detect_object_all.setPixmap(QPixmap(':/image/icons/All_nums.png'))
        self.label_detect_object_all.setScaledContents(True)
        self.label_camera_select.setPixmap(QPixmap(':/image/icons/camera_start.png'))
        self.label_camera_select.setScaledContents(True)
        self.label_camera_detect.setPixmap(QPixmap(':/image/icons/camera_detect.png'))
        self.label_camera_detect.setScaledContents(True)
        self.label_camera_stop.setPixmap(QPixmap(':/image/icons/camera_stop.png'))
        self.label_camera_stop.setScaledContents(True)
        self.input.setPixmap(QPixmap())
        self.input.setScaledContents(True)
        # self.output.setPixmap(QPixmap("input.png"))
        # self.output.setScaledContents(True)
        # images/pexels-bri-schneiter-346529.jpg
        self.label_main.setPixmap(QPixmap("ui_images/bg.png"))
        self.label_main.setScaledContents(True)

        # self.button.setStyleSheet("QPushButton:pressed { background-color: red; }")
        # self.button.clicked.connect(self.changeColor)
        self.button_weight_select.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_weight_init.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_detect.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_export.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.buttton_image_select.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_show.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_stop.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_detect.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_export.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_stop.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_suspend.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.buttton_video_select.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_camera_detect.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_camera_start.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_camera_stop.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_widget = LoginWidget()
    window = MainWindow()
    window.setWindowTitle('基于深度学习的舰船检测系统')
    window.setFixedSize(window.size())
    window.hide()
    login_widget.login_success_signal.connect(lambda: (login_widget.close(), window.show()))
    login_widget.show()
    app.exec()