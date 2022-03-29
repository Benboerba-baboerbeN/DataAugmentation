import math
import os
import pathlib
import random
import time
import cv2
import numpy as np
import ttkbootstrap as ttk

from queue import Queue
from tkinter import StringVar
from tkinter.filedialog import askdirectory
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from ttkbootstrap.constants import *

path_in = ''
path_out = ''


class DataAugmentation(ttk.Frame):
    queue = Queue()

    def __init__(self, master):
        global path_in
        self.R = RandomErasing(p=1)
        super().__init__(master, padding=15)
        self.pack(fill=BOTH, expand=YES)

        _path = pathlib.Path().absolute().as_posix()
        self.path_var = ttk.StringVar(value='输入文件夹')
        self.path_var_out = ttk.StringVar(value='输出文件夹')

        option_text = "地址选择"
        self.option_lf = ttk.Labelframe(self,
                                        text=option_text,
                                        padding=15)  # 创建第一个框架
        self.option_lf.pack(fill=X, expand=YES, anchor=N)

        self.create_path_row()  # 创建第一个框架下第一行输入
        self.output_path_row()  # 创建第一个框架下第二行输出
        option_text = "图像扩增"
        self.option_lf = ttk.Labelframe(self,
                                        text=option_text,
                                        padding=15)  # 创建第二个框架
        self.option_lf.pack(fill=X, expand=YES, anchor=N)
        self.create_combobox()  # 创建第二个框架下的内容

    def create_combobox(self):
        global cbo, cbo_2, warr, cb
        warr = StringVar()
        warr.set('')
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row,
                             text="扩增方法",
                             width=8)
        path_lbl.pack(side=LEFT, padx=(15, 0))  # 创建标签
        cbo = ttk.Combobox(
            path_row,
            values=['Resize-------调整大小', 'Rotate-------旋转图片', 'Convert------格式转换', 'Flip-----------翻转图片',
                    'Contrast-----对比度', 'Color---------饱和度', 'Brightness---亮度', 'Sharpness---锐度',
                    'RdmErasing-随机擦除',
                    'GNoise------高斯噪声',
                    'SpNoise-----椒盐噪声']
        )
        cbo.current(0)  # 设置cbo显示内容为第0
        cbo.pack(side=LEFT, padx=(0, 0))  # 创建第一个下拉菜单cbo
        cbo.bind('<<ComboboxSelected>>', model)  # cbo选择后自动执行model
        # browse_btn = ttk.Button(
        #     master=path_row,
        #     text='确认',
        #     command=model,
        #     width=8
        # )
        # browse_btn.pack(side=LEFT, padx=5)
        path_lbl = ttk.Label(path_row, text="方法细节", width=8)
        path_lbl.pack(side=LEFT, padx=(15, 0))  # 创建标签
        cbo_2 = ttk.Combobox(
            path_row,
            values=('100x100', '300x300', '600x600', '1920x1080', '2560x1440')
        )
        cbo_2.current(0)  # 设置cbo2显示内容为第0
        cbo_2.pack(side=LEFT, padx=(0, 0))  # 创建第二个下拉菜单cbo2
        browse_btn = ttk.Button(
            master=path_row,
            text="确认",
            command=self.model_2,  # 确认按钮执行model_2
            width=8
        )
        browse_btn.pack(side=LEFT, padx=(8, 5))  # 创建确认按钮
        label_version = ttk.Label(text="DataAugmentation Version 2",
                                  bootstyle=DANGER,
                                  width=30)
        label_version.pack(side=RIGHT, padx=(15, 0))  # 创建标签
        path_lbl_1 = ttk.Label(self, textvariable=warr, bootstyle=DANGER, width=8)
        path_lbl_1.pack(fill=X)  # 创建标签
        cb = ttk.Checkbutton(
            text='显示输出图像',
            variable='op1'
        )
        cb.pack(fill=X, padx=(45, 0), pady=(0, 8))  # 创建选择框
        self.setvar('op1', '0')  # 设置选择框默认为关闭

    def model_2(self):  # 确认按钮执行任务
        global cbo, cbo_2, path_ent_in, path_ent_out, cb, warr, cb2
        localtime = time.asctime(time.localtime(time.time()))
        temp = cbo.get()  # 获取cbo下拉菜单内容
        var_b2 = cbo_2.get()  # 获取cbo2下拉菜单内容
        pathIn = path_ent_in.get()  # 获取输入文件夹路径
        pathOut = path_ent_out.get()  # 获取输出文件夹路径
        # try:
        #     warr.set('输出成功: ' + localtime)
        if cb2.getvar('op2') == '1':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.png'):
                    i = Image.open(f)
                    fn, fext = os.path.splitext(f)
                    i = i.convert('RGB')
                    i.save('{}/{}.jpg'.format(pathIn.replace("\\", "\\\\"), fn))  # 重命名改后缀

        if temp == 'Resize-------调整大小':
            num1, num2 = strxint(var_b2)
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)  # 切分文件名
                    i.thumbnail((num1, num2))
                    if cb.getvar('op1') == '1':  # 如果cb选项框为1 则展示输出的图像
                        plt.imshow(i)
                        plt.show()
                    i.save('{}/{}_{}x{}{}'.format(pathOut.replace("\\", "\\\\"), fn, num1, num2, fext))  # 保存扩增后的文件
        elif temp == 'Rotate-------旋转图片':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)
                    # i.rotate(str2int(var_b2)).show()
                    if cb.getvar('op1') == '1':
                        plt.imshow(i.rotate(str2int(var_b2)))
                        plt.show()
                    i.rotate(str2int(var_b2)).save(
                        '{}/{}_Rotate_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, str2int(var_b2), fext))
        elif temp == 'Convert------格式转换':
            for f in os.listdir('.'):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)
                    if cb.getvar('op1') == '1':
                        plt.imshow(i.convert(mode=var_b2))
                        plt.show()
                    i.convert(mode=var_b2).save(
                        '{}/{}_Convert_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, var_b2, fext))
                    # i.convert(mode=var_b2).show()
        elif temp == 'Flip-----------翻转图片':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)
                    if var_b2 == 'LEFT_RIGHT':
                        if cb.getvar('op1') == '1':
                            plt.imshow(i.transpose(Image.FLIP_TOP_BOTTOM))
                            plt.show()
                        i.transpose(Image.FLIP_LEFT_RIGHT).save(
                            '{}/{}_FLIP_LR{}'.format(pathOut.replace("\\", "\\\\"), fn, fext))
                    elif var_b2 == 'TOP_BOTTOM':
                        if cb.getvar('op1') == '1':
                            plt.imshow(i.transpose(Image.FLIP_TOP_BOTTOM))
                            plt.show()
                        i.transpose(Image.FLIP_TOP_BOTTOM).save(
                            '{}/{}_FLIP_TB{}'.format(pathOut.replace("\\", "\\\\"), fn, fext))
        elif temp == 'Contrast-----对比度':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)
                    enh = ImageEnhance.Contrast(i)
                    if cb.getvar('op1') == '1':
                        plt.imshow(enh.enhance(stradjustint(var_b2)))
                        plt.show()
                    enh.enhance(stradjustint(var_b2)).save(
                        '{}/{}_Contrast_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, stradjustint(var_b2), fext))
        elif temp == 'Color---------饱和度':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)
                    enh = ImageEnhance.Color(i)
                    if cb.getvar('op1') == '1':
                        plt.imshow(enh.enhance(stradjustint(var_b2)))
                        plt.show()
                    enh.enhance(stradjustint(var_b2)).save(
                        '{}/{}_Color_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, stradjustint(var_b2), fext))
        elif temp == 'Brightness---亮度':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)
                    enh = ImageEnhance.Brightness(i)
                    if cb.getvar('op1') == '1':
                        plt.imshow(enh.enhance(stradjustint(var_b2)))
                        plt.show()
                    enh.enhance(stradjustint(var_b2)).save(
                        '{}/{}_Brightness_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, stradjustint(var_b2),
                                                       fext))
        elif temp == 'Sharpness---锐度':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = Image.open(os.path.join(pathIn.replace("\\", "\\\\"), f))
                    fn, fext = os.path.splitext(f)
                    enh = ImageEnhance.Sharpness(i)
                    if cb.getvar('op1') == '1':
                        plt.imshow(enh.enhance(stradjustint(var_b2)))
                        plt.show()
                    enh.enhance(stradjustint(var_b2)).save(
                        '{}/{}_Sharpness_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, stradjustint(var_b2),
                                                      fext))
        elif temp == 'RdmErasing-随机擦除':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = cv2.imread(os.path.join(pathIn.replace("\\", "\\\\"), f))  # 不支持中文路径
                    fn, fext = os.path.splitext(f)
                    for x in range(stradjustint(var_b2)):
                        img1 = self.R(i.copy())
                        cv2.waitKey(1000)
                        if cb.getvar('op1') == '1':
                            cv2.imshow("test", img1)
                        cv2.imwrite('{}/{}_RdmErasing_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, x, fext),
                                    img1)
        elif temp == 'GNoise------高斯噪声':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    i = cv2.imread(os.path.join(pathIn.replace("\\", "\\\\"), f))  # 不支持中文路径
                    fn, fext = os.path.splitext(f)
                    img1 = np.zeros(i.shape, np.uint8)
                    thres = 1 - stradjustint(var_b2)
                    for x in range(i.shape[0]):
                        for j in range(i.shape[1]):
                            rdn = random.random()
                            if rdn < float(var_b2):
                                img1[x][j] = 0
                            elif rdn > thres:
                                img1[x][j] = 255
                            else:
                                img1[x][j] = i[x][j]
                    if cb.getvar('op1') == '1':
                        plt.imshow(img1)
                        plt.show()
                    cv2.imwrite('{}/{}_GNoise_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, float(var_b2), fext),
                                img1)
        elif temp == 'SpNoise-----椒盐噪声':
            for f in os.listdir(pathIn.replace("\\", "\\\\")):
                if f.endswith('.jpg'):
                    mean = 0
                    i = cv2.imread(os.path.join(pathIn.replace("\\", "\\\\"), f))  # 不支持中文路径
                    fn, fext = os.path.splitext(f)
                    i = np.array(i / 255, dtype=float)
                    noise = np.random.normal(mean, float(var_b2) ** 0.5, i.shape)
                    img1 = i + noise
                    if img1.min() < 0:
                        low_clip = -1.
                    else:
                        low_clip = 0.
                    img1 = np.clip(img1, low_clip, 1.0)
                    img1 = np.uint8(img1 * 255)
                    if cb.getvar('op1') == '1':
                        plt.imshow(img1)
                        plt.show()
                    cv2.imwrite('{}/{}_SpNoise_{}{}'.format(pathOut.replace("\\", "\\\\"), fn, float(var_b2), fext),
                                img1)
        # except:
        #     warr.set('请输入正确的文件路径')

    def create_path_row(self):
        global path_ent_in
        path_row = ttk.Frame(self.option_lf)
        path_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row, text="Path in ", width=9)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        # path_ent_in = ttk.Entry(path_row, textvariable=self.path_var)
        path_ent_in = ttk.Entry(path_row, textvariable=self.path_var)
        path_ent_in.pack(side=LEFT, fill=X, expand=YES, padx=5)
        browse_btn = ttk.Button(
            master=path_row,
            text="浏览",
            command=self.on_browse,
            width=8
        )
        browse_btn.pack(side=LEFT, padx=5)

    def output_path_row(self):
        global path_ent_out, cb2
        path_row_out = ttk.Frame(self.option_lf)
        path_row_out.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(path_row_out, text="Path out", width=9)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        path_ent_out = ttk.Entry(path_row_out, textvariable=self.path_var_out)
        path_ent_out.pack(side=LEFT, fill=X, expand=YES, padx=5, pady=10)
        browse_btn = ttk.Button(
            master=path_row_out,
            text="浏览",
            command=self.out_browse,
            width=8
        )
        browse_btn.pack(side=LEFT, padx=5)
        path_row_cb = ttk.Frame(self.option_lf)
        path_row_cb.pack(fill=X, expand=YES)
        cb2 = ttk.Checkbutton(
            path_row_cb,
            text='PNG图片扩增',
            variable='op2'
        )
        cb2.pack(fill=X, padx=(20, 0), pady=(8, 0))  # 创建选择框
        self.setvar('op2', '0')  # 设置选择框默认为关闭

    def on_browse(self):
        path = askdirectory(title="Browse directory")
        if path:
            self.path_var.set(path)
            global path_in
            path_in = path

    def out_browse(self):
        path = askdirectory(title="Browse directory")
        if path:
            self.path_var_out.set(path)
            global path_out
            path_out = path


def strxint(s):  # 解析AxB
    try:
        return int(s)
    except:
        num = 0
        for i in range(len(s)):
            if s[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                num = num * 10 + int(s[i])
            elif s[i] == 'x':
                num1 = num
                num = 0

        return num1, num


def str2int(s):  # str转int
    try:
        return int(s)
    except:
        if s[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            num = 0
            for i in range(len(s)):
                if s[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    num = num * 10 + int(s[i])
                else:
                    return num
        else:
            return 0


def model(x):
    global cbo, cbo_2
    temp = cbo.get()
    if temp == 'Resize-------调整大小':
        cbo_2.config(values=('100x100', '300x300', '600x600', '1920x1080', '2560x1440'))
        cbo_2.current(0)
    elif temp == 'Rotate-------旋转图片':
        cbo_2.config(values=(
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
            '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
            '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74',
            '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92',
            '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108',
            '109',
            '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124',
            '125',
            '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
            '141',
            '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156',
            '157',
            '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172',
            '173',
            '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188',
            '189',
            '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204',
            '205',
            '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220',
            '221',
            '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236',
            '237',
            '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
            '253',
            '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268',
            '269',
            '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284',
            '285',
            '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300',
            '301',
            '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316',
            '317',
            '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332',
            '333',
            '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348',
            '349',
            '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360'))
        cbo_2.current(0)
    elif temp == 'Convert------格式转换':
        cbo_2.config(values=('1--------二值图像',
                             'L--------灰色图像',
                             'P--------8位彩色',
                             'RGB-----RGB彩色',
                             'RGBA---RGBA彩色',
                             'CMYK---CMYK彩色',
                             'YCbCr--YCbCr彩色',
                             'I--------32位灰色',
                             'F--------32位灰色(保留小数)'))
        cbo_2.current(0)
    elif temp == 'Flip-----------翻转图片':
        cbo_2.config(values=('LEFT_RIGHT', 'TOP_BOTTOM'))
        cbo_2.current(0)
    elif temp == 'Contrast-----对比度' or cbo.get() == 'Color---------饱和度' or cbo.get() == 'Brightness---亮度' or cbo.get() == 'Sharpness---锐度':
        cbo_2.config(values=(
            'Adjust to 0.5', 'Adjust to 0.7', 'Adjust to 0.9', 'Adjust to 1.1', 'Adjust to 1.3', 'Adjust to 1.5'))
        cbo_2.current(0)
    elif temp == 'RdmErasing-随机擦除':
        cbo_2.config(values=(
            '输出1张', '输出2张', '输出3张', '输出4张', '输出5张', '输出6张', '输出7张', '输出8张', '输出9张', '输出10张', '输出11张', '输出12张', '输出13张',
            '输出14张', '输出15张', '输出16张', '输出17张', '输出18张', '输出19张', '输出20张'))
        cbo_2.current(0)
    elif temp == 'GNoise------高斯噪声':
        cbo_2.config(values=(
        '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10', '0.11', '0.12', '0.13', '0.14',
        '0.15', '0.16', '0.17', '0.18', '0.19', '0.20', '0.25', '0.30', '0.40', '0.50'))
        cbo_2.current(0)
    elif temp == 'SpNoise-----椒盐噪声':
        cbo_2.config(values=(
        '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10', '0.11', '0.12', '0.13', '0.14',
        '0.15', '0.16', '0.17', '0.18', '0.19', '0.20', '0.25', '0.30', '0.40', '0.50'))
        cbo_2.current(0)


def stradjustint(s):  # Adjust to 0.5 转0.5
    try:
        return int(s)
    except:

        num = 0
        decimal = 0
        x = 0
        a = 1
        for i in range(len(s)):
            if s[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] and x != 1:
                num = num * 10 + int(s[i])
            if s[i] == '.':
                x = 1
            if s[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] and x == 1:
                decimal = decimal * 10 + int(s[i])
        b = decimal
        while b / 10 > 1:
            b = b / 10
            a = a + 1
        c = 0.1
        for i in range(a - 1):
            c = c * 100 * 0.1 / 100
        return num + decimal * c


class RandomErasing:

    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3):

        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1 / r1)

    def __call__(self, img):
        assert len(img.shape) == 3

        if random.random() > self.p:
            return img

        else:
            while True:
                Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r)
                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))
                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])
                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))

                    return img


if __name__ == '__main__':
    app = ttk.Window("Data Augmentation", "journal")
    DataAugmentation(app)
    app.mainloop()
