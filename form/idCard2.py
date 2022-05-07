import cv2
import base64
import os
import sys
import copy

sys.path.insert(0, ".")

from tools.infer.utility import parse_args
from tools.infer.predict_det import TextDetector


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

def merge_configs():
    # deafult cfg
    backup_argv = copy.deepcopy(sys.argv)
    sys.argv = sys.argv[:1]
    cfg = parse_args()

    update_cfg_map = vars(read_params())

    for key in update_cfg_map:
        cfg.__setattr__(key, update_cfg_map[key])

    sys.argv = copy.deepcopy(backup_argv)
    return cfg

class Config(object):
    pass


def read_params():
    cfg = Config()

    # params for text detector
    cfg.det_algorithm = "DB"
    # cfg.det_model_dir = "./inference/idCard2/ch_db_mv3_sfz/"
    cfg.det_model_dir = "./output_inference/ch_db_mv3_sfz/"
    # cfg.det_model_dir = "./inference/idCard2/det_r50_vd_db_sfz/"
    cfg.det_resize_long = 960

    # DB parmas
    cfg.det_db_thresh = 0.3
    cfg.det_db_box_thresh = 0.5
    cfg.det_db_unclip_ratio = 2.0
    cfg.use_dilation = False
    cfg.det_db_score_mode = "fast"

    # #EAST parmas
    # cfg.det_east_score_thresh = 0.8
    # cfg.det_east_cover_thresh = 0.1
    # cfg.det_east_nms_thresh = 0.2

    cfg.use_pdserving = False
    cfg.use_tensorrt = False

    return cfg


#  身份证识别
class IdCard(object):
    # 初始化
    def __init__(self, text_system, args):
        self.text_system = text_system
        self.score = 0.5
        # 实例分割
        # if args["idCard"]["accuracy"] == "accurate":
        #     self.directory = "./inference/idCard/cascade_mask_rcnn_mobilenetv3_fpn_1x/"
        # else:
        #     self.directory = "./inference/idCard/mask_rcnn_mobilenetv3_fpn_1x/"

        cfg = merge_configs()

        cfg.use_gpu = args["use_gpu"]
        if args["use_gpu"]:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", args["use_gpu"])
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True
        cfg.enable_mkldnn = args["enable_mkldnn"]

        self.model = TextDetector(cfg)

    def idCard(self, img):
        dt_boxes, dt_labels, elapse = self.model(img, cls=True)

        print(dt_boxes)
        print(dt_labels)
        new_results = {}

        categorys = ["name", "sex", "nation", "birthday", "address", "number"]

        if str(new_results) == '{}':
            return ""

        return new_results

    def __call__(self, img=None):
        return self.idCard(img)


if __name__ == '__main__':
    # from deploy.hubserving.ocr_system.module import OCRSystem
    # text_system = OCRSystem({
    #     "use_gpu": True,
    #     "enable_mkldnn": False
    # }).text_sys

    module = IdCard(None, {
        "use_gpu": True,
        "enable_mkldnn": False
    })
    image = cv2.imread('E:\\dataset\\shenfenzheng\\images\\JPEGImages\\001.png')
    res = module(img=image)
