import cv2
import os
import numpy as np
from ExtrinsicCalibration import ExCalibrator
from IntrinsicCalibration import InCalibrator, CalibMode
from SurroundBirdEyeView import BevGenerator

# TODO: 在这里选择 case
# CASE_NAME = 'default'
# CASE_NAME = 'gray'
# CASE_NAME = 'gray_same'
CASE_NAME = 'rgb_same'

CASE_DICT = {
    'rgb_same': {
        'INPUT_PATH':   './IntrinsicCalibration/data/rgb_same/',
        'IMAGE_FILE':   './IntrinsicCalibration/data/rgb_same/IMG_0001.jpg',
        # 'img_src_back': './ExtrinsicCalibration/data/default/img_src_back.jpg',  #TODO
        # 'img_dst_back': './ExtrinsicCalibration/data/default/img_dst_back.jpg',  #TODO
        'SUBPIX_REGION': 20,
        'SQUARE_SIZE': 80,
        
        'BOARD_WIDTH': 9,
        'BOARD_HEIGHT': 6,
        'IMAGE_FILE': 'IMG_',
#         Camera Matrix is: [[978.8593165475615, 0.0, 1357.7303197656022], [0.0, 978.3270054737604, 947.4822226546848], [0.0, 0.0, 1.0]]
# Distortion Coefficient is: [[0.3036094923652017], [0.4859345373456774], [-0.3881486035004563], [0.0984832229319884]]

    },
    'gray_same': {
        'INPUT_PATH':   './IntrinsicCalibration/data/gray_same/',
        'IMAGE_FILE':   './IntrinsicCalibration/data/gray_same/g001.png',
        # 'img_src_back': './ExtrinsicCalibration/data/default/img_src_back.jpg',  #TODO
        # 'img_dst_back': './ExtrinsicCalibration/data/default/img_dst_back.jpg',  #TODO

        'BOARD_WIDTH': 9,
        'BOARD_HEIGHT': 6,
        'IMAGE_FILE': 'g',
        
        ## LiDAR 内参输出：
        # Camera Matrix is: 
        # [[348.6070847974017, 0.0, 381.7880841844476], 
        #  [0.0, 348.6942253301466, 303.2254092177489],
        #  [0.0, 0.0, 1.0]]
        # Distortion Coefficient is: [[0.0036238320266094304], [-0.0022551352630669685], [-0.004834645558475566], [0.0022603433251687528]]
        ##
    },
    'default': {
        'INPUT_PATH':   './IntrinsicCalibration/data/default/',
        'IMAGE_FILE':   './IntrinsicCalibration/data/default/img_raw0.jpg',
        'img_src_back': './ExtrinsicCalibration/data/default/img_src_back.jpg',
        'img_dst_back': './ExtrinsicCalibration/data/default/img_dst_back.jpg',

        'BOARD_WIDTH': 7,
        'BOARD_HEIGHT': 6,
        'IMAGE_FILE': 'img_raw',
    },
}


def runInCalib_2():
    print("Intrinsic Calibration......")
    
    ## 参数（见 intrinsicCalib.py 开头）
    args = InCalibrator.get_args()                      # 获取内参标定 args 参数
    for key in CASE_DICT[CASE_NAME].keys():
        if hasattr(args, key):  # 没有的属性就采用默认值
            setattr(args, key, CASE_DICT[CASE_NAME][key])

    ##
    
    calibrator = InCalibrator('fisheye')                # 初始化内参标定器
    calib = CalibMode(calibrator, 'image', 'auto')      # 选择标定模式
    result = calib()                                    # 开始标定

    print("Camera Matrix is: {}".format(result.camera_mat.tolist()))
    print("Distortion Coefficient is: {}".format(result.dist_coeff.tolist()))
    print("Reprojection Error is: {}".format(np.mean(result.reproj_err)))

    raw_frame = cv2.imread(CASE_DICT[CASE_NAME]['IMAGE_FILE'])
    # calibrator.draw_corners(raw_frame)                  # 画出角点
    cv2.imshow("Raw Image", raw_frame)
    undist_img = calibrator.undistort(raw_frame)        # 使用 undistort 方法得到去畸变图像
    cv2.imshow("Undistort Image", undist_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def runExCalib():
    print("Extrinsic Calibration ......")
    exCalib = ExCalibrator()                            # 初始化外参标定器

    src_raw = cv2.imread(CASE_DICT[CASE_NAME]['img_src_back'])
    dst_raw = cv2.imread(CASE_DICT[CASE_NAME]['img_dst_back'])

    homography = exCalib(src_raw, dst_raw)              # 输入对应的两张去畸变图像 得到单应性矩阵
    print("Homography Matrix is:")
    print(homography.tolist())

    src_warp = exCalib.warp()                           # 使用warp方法得到原始图像的变换结果

    cv2.namedWindow("Source View", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Source View", src_raw)
    cv2.namedWindow("Destination View", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Destination View", dst_raw)
    cv2.namedWindow("Warped Source View", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Warped Source View", src_warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def runBEV():
    print("Generating Surround BEV ......")
    front = cv2.imread('./SurroundBirdEyeView/data/front/front.jpg')
    back = cv2.imread('./SurroundBirdEyeView/data/back/back.jpg')
    left = cv2.imread('./SurroundBirdEyeView/data/left/left.jpg')
    right = cv2.imread('./SurroundBirdEyeView/data/right/right.jpg')

    args = BevGenerator.get_args()                      # 获取环视鸟瞰 args 参数
    args.CAR_WIDTH = 200
    args.CAR_HEIGHT = 350                               # 修改为新的参数

    bev = BevGenerator(blend=True, balance=True)        # 初始化环视鸟瞰图生成器
    surround = bev(front, back, left, right)            # 输入前后左右四张原始相机图像 得到拼接后的鸟瞰图

    cv2.namedWindow('surround', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('surround', surround)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # runInCalib_1()
    runInCalib_2()
    # runExCalib()
    # runBEV()

if __name__ == '__main__':
    main()

