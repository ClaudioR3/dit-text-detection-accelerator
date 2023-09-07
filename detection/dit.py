from ditod import add_vit_config, MyTrainer
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor, default_setup
from detectron2.structures import BoxMode

IMAGE_URLS = []
IMAGE_ANNOTATIONS = []

def init_detector(config_file, checkpoint):
    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(['MODEL.WEIGHTS', checkpoint])
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: set classes
    #md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    #md.set(thing_classes=["text"])

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    return predictor
    
def inference_detector(predictor, img):
    # run inference
    output = predictor(img)["instances"]
    fields = output.to("cpu").get_fields()

    ## 'pred_boxes', 'scores'
    results = {
        'boxes': [x.tolist() for x in fields['pred_boxes']],
        'scores': fields['scores'].tolist()
    }
    return results

def hnl_dataset_function(): 
    data = []
    for i, (url, annotations) in enumerate(zip(IMAGE_URLS, IMAGE_ANNOTATIONS)):
        tmp = {}
        tmp['file_name'] = url
        tmp['height'] = annotations[0]['original_height']
        tmp['width'] =  annotations[0]['original_width']
        tmp['image_id'] = i
        tmp['annotations'] = []

        for anno in annotations:
            if anno['value']['rectanglelabels'] == 'text':
                continue
            elif anno['value']['rotation'] != 0:
                continue
            else:
                x = anno['value']['x'] * tmp['width'] / 100
                y = anno['value']['y'] * tmp['height'] / 100
                width = anno['value']['width'] * tmp['width'] / 100
                height = anno['value']['height'] * tmp['height'] / 100
                bbox = (x, y, width, height)
                tmp['annotations'].append({
                    'bbox': bbox,
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': 0,
                    'segmentation': {}
                })

        data.append(tmp)
    #print(data)
    return data

def prepare_dataset(image_urls, image_annotations):
    """
    Register custom dataset as DatasetCatalog in Detectron2 format
        Args:
            image_urls (list): List of images' urls.
            image_annotations (list): List of images' annotations.

        Returns:
            None
    """
    global IMAGE_URLS, IMAGE_ANNOTATIONS 
    IMAGE_URLS = image_urls
    IMAGE_ANNOTATIONS = image_annotations
    DatasetCatalog.register("hnl_train", hnl_dataset_function)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def train_detector(config_file, checkpoint, output_dir = 'training'):
    args = Argument(config_file, ['MODEL.WEIGHTS', checkpoint, 'OUTPUT_DIR', output_dir])
    # Step 1: instantiate config
    cfg = setup(args)
    # default_setup(cfg, args)
    # Step 2: training phase
    trainer = MyTrainer(cfg)
    trainer.resume_or_load()
    return trainer.train()

class Argument(object):
    def __init__(self, config_file, opts) -> None:
        self.config_file = config_file
        self.opts = opts

def save_detector(model, checkpoint = 'last_model.pth'):
    torch.save(model, checkpoint)