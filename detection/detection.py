import os
import logging
import boto3
import io
import json
import cv2
import urllib
import numpy as np  
import requests
import shutil

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, DATA_UNDEFINED_NAME, is_skipped
# from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse

from dit import init_detector, inference_detector, train_detector, prepare_dataset, save_detector


HOSTNAME = os.environ['HOSTNAME']
API_KEY = os.environ['API_KEY']
TRAINING_PATH = '/data/training'
MODEL = init_detector(os.environ['config_file'], os.environ['checkpoint_file'])

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')

logger = logging.getLogger(__name__)


class Detection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, labels_file=None, score_threshold=0.3, **kwargs):
        """
        Load Detection model from config and checkpoint into memory.

        Optionally set mappings from COCO classes to target labels
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param kwargs:
        """
        super(Detection, self).__init__(**kwargs)
        self.model_version=0
        self.labels_file = labels_file
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}
        logger.info(f'{self.__class__.__name__} defines LABEL MAP')

        #self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
        #    self.parsed_label_config, 'Rectangle', 'Image')
        self.from_name, self.to_name, self.value = 'label', 'image', 'image'
        logger.info(f'{self.__class__.__name__} defines FROM NAME: {self.from_name}, TO NAME: {self.to_name}, VALUE: {self.value}')
        #print('PARSED LABEL CONFIG {}'.format(self.parsed_label_config))
        schema = list(self.parsed_label_config.values())[0]
        
        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name
        # define model
        self.model = MODEL
        logger.info(f'{self.__class__.__name__} defines MODEL ({self.model_version})')
        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://') or image_url.startswith('s3: //'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3',
                aws_access_key_id = os.environ['ACCESS_KEY'],
                aws_secret_access_key = os.environ['SECRET_KEY'],
                endpoint_url = os.environ['ENDPOINT_URL'])
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def _get_annotated_dataset(self, project_id):
        """Retrieve annotated data from Label Studio API"""
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {download_url}, "
                            f"response status_code = {response.status_code}")
        return json.loads(response.content)

    def _check_updated_model(self):
        # if self.train_output:
        #     if "model_path" in self.train_output and "event" in self.train_output and "model_version" in self.train_output:
        #         if self.train_output['event'] == 'PROJECT_UPDATED' and self.train_output['model_version'] > self.model_version:
        #             print(self.train_output)
        #             self.model =  init_detector(os.environ['config_file'], self.train_output['model_path'])
        #             self.model_version = self.train_output['model_version']
        model_path = os.path.join(TRAINING_PATH, 'model_final.pth')
        if os.path.exists(model_path):
            self.model = init_detector(os.environ['config_file'], model_path)

    def predict(self, tasks, **kwargs):
        urls = [task['data'][self.value] for task in tasks]
        print(f'Predicting {len(urls)} images')

        # Check and Load model with an updated version
        #self._check_updated_model()

        # predict bounding boxes 
        predictions = []
        for task in tasks:
            img_url = self._get_image_url(task)
            # image_path = self.get_local_path(image_url)
            img = get_image_from_url(img_url)
            img_height, img_width, _ = img.shape
            img = preprocessing(img)
            model_results = inference_detector(self.model, img)
            # define Label Studio's predictions
            results = []
            all_scores = []
            for bbox, score in zip(model_results['boxes'], model_results['scores']):
                if score < self.score_thresh:
                    continue
                x, y, xmax, ymax = bbox
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'x': x / img_width * 100,
                        'y': y / img_height * 100,
                        'width': (xmax - x) / img_width * 100,
                        'height': (ymax - y) / img_height * 100,
                        'rotation': 0,
                        "rectanglelabels": [
                                "text"
                            ]
                    },
                    'score': score
                })
                all_scores.append(score)
            avg_score = sum(all_scores) / max(len(all_scores), 1)
            predictions.append({
                'result': results,
                'score': avg_score
            })
        return predictions

    def fit(self, _, workdir=None, **kwargs):
        """
        if event == 'PROJECT_UPDATED' -> retraining with all tasks
        if event == 'ANNOTATION_UPDATED' -> transfer learning with a task
        """
        assert 'event' in kwargs
        # Step 1: Collect urls and annotations for each annotated image
        image_urls, image_annotations = [], []
        print('Collecting urls and annotations...')
        if kwargs['event'] == 'PROJECT_UPDATED': 
            # Retrieve the annotation ID from the payload of the webhook event
            # Use the ID to retrieve annotation data using the SDK or the API
            project_id = kwargs['data']['project']['id']
            tasks = self._get_annotated_dataset(project_id)
            for task in tasks:
                image_urls.append(self._get_image_url(task))
                image_annotations.append(task['annotations'][0]['result'])
            print('TODO -> Retraining...')
            #return {'model_path': os.environ['checkpoint_file']}
        elif kwargs['event'] == 'ANNOTATION_UPDATED':
            # Take url and annotation from data argument
            iurl = self._get_image_url(kwargs['data']['task'])
            image_urls.append(iurl)
            ianno = kwargs['data']['annotation']['result']
            image_annotations.append(ianno)
        
        # Step 2: Register a Dataset using Detectron2 lib
        print(f'Creating dataset with {len(image_urls)} images...')
        prepare_dataset(image_urls, image_annotations)

        # Step 3: Train model starting from pre-trained model
        print('Train model...')
        shutil.rmtree(TRAINING_PATH)
        model_path=None
        if kwargs['event'] == 'PROJECT_UPDATED': 
            model_path = os.path.join(TRAINING_PATH, 'model_final.pth')
            self.model = train_detector(
                os.environ['config_file'], 
                os.environ['checkpoint_file'],
                output_dir=TRAINING_PATH
                )
        elif kwargs['event'] == 'ANNOTATION_UPDATED':
            model_path = os.path.join(workdir, 'last_model.pth')
            self.model = train_detector(
                os.environ['config_file'], 
                model_path if os.path.exists(model_path) else os.environ['checkpoint_file'],
                output_dir=TRAINING_PATH
                )

        # Step 4: Reload parameters
        self.model = init_detector(os.environ['config_file'], os.path.join(TRAINING_PATH, 'model_final.pth'))

        # Step 5: Save model
        #print('Save model...')
        #save_detector(self.model, model_path)
        return {
            'model_path': model_path,
            'model_version': self.model_version+1,
            'event':kwargs['event']
            }

def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data

def get_image_from_url(img_url):
    # Step 1: download image
    req = urllib.request.urlopen(img_url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'
    return img

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        199, 5)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)