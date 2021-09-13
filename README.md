# Some Memos

## Library to use

https://github.com/microsoft/cameratraps

Or use ResNet??

## Classification

https://github.com/microsoft/cameratraps#classification
https://github.com/microsoft/CameraTraps/blob/master/archive/classification_marcel/TUTORIAL.md


## 환경을 갖추기는 어려운 편이다. 처음 하는 사람들에게는.

Docker로 되어있지 않아서 100% 수작업으로 setup해야함.
리눅스 환경을 권장한다.
Announcing CUDA on Windows Subsystem for Linux 2 | NVIDIA
이런것이 있기는하지만 그래도 PC는 논문도 쓰고 다른작업도 해야하기 때문에
deep learning이 도는 머신과 작업용 머신은 분리하기를 권장한다.

### venv

python 3.9를 기준으로 작성한다.

```
$ python --version
Python 3.9.6

$ python -m venv --system-site-packages ~/usr/venv-tf39
$ source ~/usr/venv-tf39/bin/activate

$ export CAMERATRAPS_DIR=`pwd`/CameraTraps
$ git clone https://github.com/Microsoft/CameraTraps.git $CAMERATRAPS_DIR
```

패키지를 설치해야 하는데...
```
$ conda env create -f ${CAMERATRAPS_DIR}/environment-classifier.yml
```

conda를 안 쓰고 하기는 조금 골치아프치만 environment-classifier.yml 파일을 참고해서 설치하면 못 할 일은 아니다.
대충 아래와 같다.
```
pip install "tensorflow>=2.3"  # 요새는 그냥 이렇게 설치하면 GPU support 들어있음. 따옴표 생략하면 안 됨.
pip install numpy
pip install pillow
pip install tqdm pycocotools
```

```
$ python
Python 3.9.6 (default, Jun 30 2021, 10:22:16)
[GCC 11.1.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
2021-09-01 18:46:21.613069: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-09-01 18:46:21.613108: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
>>>
```
`>>>` 나왔을 때 Ctrl+D 키를 누르면 나올 수 있다.
위와 같이 import에 성공하면 tensorflow 설치가 성공한 것이다.
Uhmm... Not quite. 위 메시지는 CUDA 버전이 안 맞아서 GPU 인식이 안 된 것임.
Tensorflow 설치에 대한 자세한 사항은 이 글이 다루고자 하는 일에 비하면 out of scope이다.
각자 알아서 잘 설치하자.

Dataset을 다운로드 받는 것이 이후 tutorial에 나와있는데... 그러기 위해선 azcopy 필요.
설치한다.
압축을 해제하면 azcopy란 파일이 있는데 ~/usr/venv-tf39/bin 에 복사하거나 이동하면 설치가 된다.

### Input Format

우리 입력을 detector에 맞게 변형을 가해줘야 하는데,
문제는 detector가 원하는 파일 형식을 모르겠다는 것!
일단 메타데이터부터 받아보자.

```
mkdir serengeti
cd serengeti
BASEURL=https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0
dest="SnapshotSerengeti.json.zip"
azcopy cp "${BASEURL}/SnapshotSerengeti_S1-11_v2_1.json.zip" "${dest}"
unzip -q ${dest}
```

압축을 해제해보니 JSON 파일 단 한개가 들어있다.
압축해제 전에는 용량이 180M,
용량은 무려 5.2G이다.
gzip으로 다시 묶고

```
zcat SnapshotSerengeti_S1-11_v2.1.json.gz | head -n 10000 > head.txt
zcat SnapshotSerengeti_S1-11_v2.1.json.gz | tail -n 10000 > tail.txt
```
로 구조를 엿보면

```
{
 "info": {
  "version": "2.1",
  "description": "Camera trap data from the Snapshot Serengeti program, seasons 1-11",
  "date_created": "2019",
  "contributor": "University of Minnesota Lion Center"
 },
 "categories": [
  {
   "id": 0,
   "name": "empty"
  },
  {
   "id": 1,
   "name": "human"
  },
  ...
  {
   "id": 60,
   "name": "lioncub"
  }
 ],
 "images": [
  {
   "id": "S1/B04/B04_R1/S1_B04_R1_PICT0001",
   "file_name": "S1/B04/B04_R1/S1_B04_R1_PICT0001.JPG",
   "frame_num": 1,
   "seq_id": "SER_S1#B04#1#1",
   "width": 2048,
   "height": 1536,
   "corrupt": false,
   "location": "B04",
   "seq_num_frames": 1,
   "datetime": "2010-07-18 16:26:14"
  },
  {
   "id": "S1/B04/B04_R1/S1_B04_R1_PICT0002",
   "file_name": "S1/B04/B04_R1/S1_B04_R1_PICT0002.JPG",
   "frame_num": 1,
   "seq_id": "SER_S1#B04#1#2",
   "width": 2048,
   "height": 1536,
   "corrupt": false,
   "location": "B04",
   "seq_num_frames": 1,
   "datetime": "2010-07-18 16:26:30"
  },
  ...
  {
   "sequence_level_annotation": true,
   "id": "453e191b-91f8-11e9-bd66-000d3a198845",
   "category_id": 1,
   "seq_id": "SER_S11#T11#1#131",
   "season": "SER_S11",
   "datetime": "2015-11-18 10:23:36",
   "subject_id": 31711358,
   "count": NaN,
   "standing": 0.7,
   "resting": 0.0,
   "moving": 0.4,
   "interacting": 0.0,
   "young_present": 0.0,
   "image_id": "SER_S11/T11/T11_R1/SER_S11_T11_R1_IMAG0335",
   "location": "T11"
  },
  {
   "sequence_level_annotation": true,
   "id": "453e191c-91f8-11e9-bacf-000d3a198845",
   "category_id": 1,
   "seq_id": "SER_S11#T11#1#131",
   "season": "SER_S11",
   "datetime": "2015-11-18 10:23:36",
   "subject_id": 31711358,
   "count": NaN,
   "standing": 0.7,
   "resting": 0.0,
   "moving": 0.4,
   "interacting": 0.0,
   "young_present": 0.0,
   "image_id": "SER_S11/T11/T11_R1/SER_S11_T11_R1_IMAG0336",
   "location": "T11"
  }
 ]
}
```

이미지도 있고 sequence level annotation도 있다니 경악스럽다.
이미지 그 자체만으론 쓸모가 없기 때문에 데이터에 대한 데이터, 즉 메타데이터가 필요한데 우리도 위와 비슷하게 작성을 해야한다.
메타데이터와 이미지데이터를 결합하는 스크립트가 있으니 의미를 정확히 파악하려고 하기보다는 돌리면서 파악해나가보자.

### Creating the Dataset

https://github.com/microsoft/CameraTraps/blob/master/archive/classification_marcel/TUTORIAL.md#running-the-detector
Running the detector 를 참고하면,

```
$ cd $CAMERATRAPS_DIR/data_management/databases/classification
```

이 안에 make_classification_dataset.py 스크립트가 있다.
그것을 돌리기 위해 준비를 좀 해보자.

```
GHOTI=$HOME/work/ghoti-2021
DATASET_DIR=$GHOTI/mydataset
COCO_STYLE_OUTPUT=$DATASET_DIR/cropped_coco_style
TFRECORDS_OUTPUT=$DATASET_DIR/cropped_tfrecords
mkdir -p $DATASET_DIR

CAMERTRAPS_DIR=$HOME/work/ghoti-2021/CameraTraps
MEGADETECTOR_PB="${CAMERTRAPS_DIR}/pbs/md_v4.1.0.pb"
mkdir -p "${CAMERTRAPS_DIR}/pbs"
curl https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb > ${MEGADETECTOR_PB}
```

이제 JSON 파일을 만들것이다. data/ghoti.json을 아래 내용으로 작성했다. (파일을 직접 열어보시오...)
https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format
위에서 zcat을 사용해서 구조를 엿보긴 했는데, 여기 나온 포맷과 일치한다.
자신있게 작성하면 된다.

```
{
  "info": {
    "version": "0.1",
      "description": "GHOTI",
      "date_created": "2021",
      "contributor": "Deokjin Joo, Ye-seul Kwan, Jongwoo Song, Catarina Pinho, Jody Hey and Yong-Jin Won"
  },

  "categories": [
    { "id": 0, "name": "gm_f" },
    { "id": 1, "name": "lf_m" }
  ],

  "images": [
    { "id": "08-0128", "file_name": "gm_f/08-0128.jpg" },
    { "id": "08-0311", "file_name": "lf_m/08-0311.jpg" }
  ],

  "annotations": [
    { "id": "08-0128", "image_id": "08-0128", "category_id": 0 }
    { "id": "08-0311", "image_id": "08-0311", "category_id": 1 }
  ]
}
```

대략 위와 같이 2개 sample만 작성하고 실행이 되나 안 되나, 인식이 잘 되나 안 되나부터 보는게 급선무다.

20_preproc.sh 를 작성했다. `bash 20_preproc.sh`로 실행하면 된다.

```
GHOTI=$HOME/work/ghoti-2021
cd ~/work/ghoti-2021/CameraTraps/data_management/databases/classification

python make_classification_dataset.py \
    $GHOTI/data/ghoti.json \    # input_json
    $GHOTI/data/ \                          # image_dir
    $MEGADETECTOR_PB \                       # frozen_graph
    --coco_style_output $COCO_STYLE_OUTPUT \
    --tfrecords_output $TFRECORDS_OUTPUT \
    --location_key location
```

실행이... 안된다.
코드가 깨져있어서 수정해야한다.
그럴 줄 알았다. 텐서플로우는 새 버전 나올때마다 깨진다. 쓰기 싫다. 강력히 비추하고 PyTorch를 배우는 것을 추천한다.

```
Traceback (most recent call last):
  File "/home/jdj/work/ghoti-2021/CameraTraps/data_management/databases/classification/make_classification_dataset.py", line 329, in <module>
    def run_detection(sess: tf.Session,
AttributeError: module 'tensorflow' has no attribute 'Session'
```

실제로 가보면 겨우 type annotation 때문에 일어나는 일이다.

```
def run_detection(sess: tf.Session,
```

이렇게 된걸

```
def run_detection(sess: "tf.Session",
```

이렇게 바꾸면 대충 해결된다. 저런 에러가 보일때마다 따옴표를 친다.

```
AttributeError: module 'tensorflow' has no attribute 'GraphDef'
```

GraphDef 찾는데마다

```
tf.compat.v1.GraphDef()   # -> instead of tf.GraphDef()
```

이런식으로 변경한다.
귀찮아서 TF 2.3으로 다운그레이드 하려고해도 안 된다...;;;
I hate TF.

```
tfrecords/utils/create_tfrecords.py", line 178, in __init__
    self._sess = tf.Session()
AttributeError: module 'tensorflow' has no attribute 'Session'
```

이제 실행되는 부분에서 tf.Session() 에러가 나오기 시작한다.

make_classification_dataset.py 파일 맨 앞부분에다가

```
import numpy as np
from PIL import Image
import pycocotools.coco
import tensorflow as tf
import tqdm

tf.compat.v1.disable_eager_execution()  # 이거 삽입
```

에러가 나는 족족 tf.Session() 대신에 tf.compat.v1.Session() 을 사용하면 된다.
귀찮다. 광범위하다.

```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

import tensorflow as tf 가 보이는 족족 위처럼 바꿔 넣어서 한방에 해결하자.

### Iteration

실행이 안 되는 문제는 해결되었는데 이번엔 JSON파일에서 이미지 각 항목에 "location"이라는 key가 없다고 불평한다.
KeyError: 'location'
이런 에러다. 달아주었다.

```
    { "id": "08-0128", "file_name": "gm_f/08-0128.jpg", "location": "dummy" },
```

이런 식으로 수정되었다.

To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: No images were written to tfrecords!
WARNING: No images were written to tfrecords!

이렇게 나온다...
알고보니 annotations를 annotation이라고 불러서 문제가 생긴 것이었다.

```
  File "/home/jdj/work/ghoti-2021/CameraTraps/data_management/databases/classification/make_classification_dataset.py", line 509, in save_outputs
    imsize = [cur_image['height'], cur_image['width']]
```

이번엔 이미지에 width height를 요구하네.
이미지 샘플을 2개만 넣기를 잘 했다.

~/work/ghoti-2021/mydataset/cropped_coco_style 에 들어가보면 파일들이 복사가 되어있고 train.json 이라든지 test.json이라든지 파일이 정확히 들어간 것으로 보인다.
json파일들의 경우 내용물을 텍스트 에디터로 보고 정확한지 판단을 잘 해야 함.

cropped_tfrecords의 경우는 좀 애매하다.

```
Use tf.gfile.GFile.
WARNING: No images were written to tfrecords!
```

이런 에러가 나오면서 전처리 프로그램이 종료되었기 때문에 에러가 난 것 같이 보인다.
하지만 용량을 체크해보면 coco와 비슷한데...

```
~/work/ghoti-2021/mydataset $ du -sh *
128K    cropped_coco_style
100K    cropped_tfrecords
```

용량으로 판단해보자면 된 것 같음.

### 대량의 JSON Entry 생성

프로그래밍적으로 하는 방법이 있고 그 방법을 추천한다.
11_make_json.py 를 실행하면 된다.
그리고 preprocess 재실행.

### Dataset Statistics

```
$ ./21_dataset_stats.sh
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
Statistics of the training split:
Locations used:
[]
In total 12 classes and 0 images.
Classes with one or more images: 0
Images per class:
ID    Name            Image count
    0 gm_f                      0
    1 lf_m                      0
    2 mv_f                      0
    3 pe_m                      0
    4 pf_f                      0
    5 pg_f                      0
    6 tg_f                      0
    7 tg_m                      0
    8 tm_f                      0
    9 tm_m                      0
   10 toc_f                     0
   11 toc_m                     0


Statistics of the testing split:
Locations used:
['dummy']
In total 12 classes and 644 images.
Classes with one or more images: 12
Images per class:
ID    Name            Image count
    0 gm_f                     13
    1 lf_m                     12
    2 mv_f                     11
    3 pe_m                     37
    4 pf_f                     11
    5 pg_f                     26
    6 tg_f                    192
    7 tg_m                    100
    8 tm_f                     24
    9 tm_m                     17
   10 toc_f                   122
   11 toc_m                    79
```

문제가 있다. training set이 비어있다!
Training/test set나누는 기준이 location 값을 사용한다.
Location을 이용해서 알아서 train/test set을 우리가 손으로 나눠야 할 것으로 보인다.
Dataset 나누기를 training시에 online으로 하지 않으니 우리가 손으로 하는 것도 이상하진 않음...
11_make_json을 수정해서 location=train, location=test 양분을 했다.
물론 클래스에 대한 빈도수가 train/test 수가 같도록 신경써서 나눠야 한다. (용어는 stratify...)

### make_classfication_dataset.py 수정

location에 의해 train/test 분할을 하긴 하는데, 어떤 것이 테스트고 어떤 것이 train인지 구분을 지정할 수 없다.
스크립트에 직접 손을 대었다.

280행 근처:

```
# test_locations = sorted(
#     random.sample(locations, max(1, int(test_fraction * len(locations)))))
# JOO: train_locations = sorted(set(locations) - set(test_locations))
train_locations = ["train"]  # JOO
test_locations = ["test"]  # JOO
```

It worked!

```
Statistics of the training split:
Locations used:
['train']
In total 12 classes and 516 images.
Classes with one or more images: 12
Images per class:
ID    Name            Image count
    0 gm_f                     10
    1 lf_m                     10
    2 mv_f                      9
    3 pe_m                     28
    4 pf_f                      9
    5 pg_f                     21
    6 tg_f                    153
    7 tg_m                     81
    8 tm_f                     19
    9 tm_m                     14
   10 toc_f                    97
   11 toc_m                    65


Statistics of the testing split:
Locations used:
['test']
In total 12 classes and 128 images.
Classes with one or more images: 12
Images per class:
ID    Name            Image count
    0 gm_f                      3
    1 lf_m                      2
    2 mv_f                      2
    3 pe_m                      9
    4 pf_f                      2
    5 pg_f                      5
    6 tg_f                     39
    7 tg_m                     19
    8 tm_f                      5
    9 tm_m                      3
   10 toc_f                    25
   11 toc_m                    14
```

### Training

30_pretrained.sh

```
PRETRAINED_DIR=`pwd`/pretrained
mkdir $PRETRAINED_DIR && cd $PRETRAINED_DIR
wget -O inc4.tar.gz http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
tar xzf inc4.tar.gz
```

```
cd $CAMERATRAPS_DIR/classification/tf-slim/datasets/
cp cct.py serengeti.py
```

Uhmmm... 공식 문서가 out of date이다.
cct.py 그런거 없다.
CameraTraps/classification/train_classifier.py 이 파일은 있다.
위에서 inc4.tar.gz를 받았지만 결정적으로 train_classifier.py라든지, 내용물을 보면 EfficientNet으로 된 것으로 보인다.
Inception v4는 무시하고 (ResNet이 대세인데 inc4라니 마음에 들지 않았다.)
train_classifier.py 설명을 보면 pretrained flag가 있다든지하여,
가이드 무시하고 script에 집중하는 것이 좋아보인다.
무려 torch로 되어있다 ㅎㄷㄷ

### Retry

```
FileNotFoundError: [Errno 2] No such file or directory: 'mydataset/cropped_coco_style/classification_ds.csv'
```

이런 에러가 난다. Dataset 만드는 것도 잘 안 된다.
README로 추정해보면 COCO dataset style이면 다 되는 것 같은데 뭐지?
코드 전체를 검색해보면 classification_ds.csv가 등장하는 곳은 train_classifier.py 에서뿐이다.
train_classifier.py는 아무래도 이 git repository에서 실험적으로 누가 만들고 관리하지 않는 파일 같아보인다.
train_classifier_tf로 시도해도 마찬가지다.

거의 포기했는데 "CameraTraps classification_ds.csv"로 구글링해보니 classification README가 검색된다.
https://github.com/microsoft/CameraTraps/blob/master/classification/README.md

또 알아듣기 힘든 말만 잔뜩 나온다.
웬만하면 남의 연구용코드와는 상종하지 않는 편이 좋다.
(train_classifier_tf에 대한 언급도 나온다. 결국 EfficientNet때문에 PyTorch로 선회했다는 내용이 나온다.
See? Tensorflow sucks!)

정말 하기 싫은데 마저 해보자.

```
        dataset_csv_path: str, path to CSV file with columns
            ['dataset', 'location', 'label'], where label is a comma-delimited
            list of labels
```

이런 글을 힌트로하여

run_ghoti/classification_ds.csv:

```
dataset,location,label,confidence
ghoti,/home/jdj/work/ghoti-2021/mydataset/cropped_coco_style,ghoti,1.0
```

이렇게 작성함.

```
$ ./31_train.sh
/home/jdj/usr/venv-3.9/lib/python3.9/site-packages/setuptools/distutils_patch.py:25: UserWarning: Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesirable behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.
  warnings.warn(
Created logdir: /home/jdj/work/ghoti-2021/run_ghoti/20210912_211827
Creating dataloaders
Traceback (most recent call last):
  File "/usr/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/jdj/work/ghoti-2021/CameraTraps/classification/train_classifier.py", line 797, in <module>
    main(dataset_dir=args.dataset_dir,
  File "/home/jdj/work/ghoti-2021/CameraTraps/classification/train_classifier.py", line 337, in main
    loaders, label_names = create_dataloaders(
  File "/home/jdj/work/ghoti-2021/CameraTraps/classification/train_classifier.py", line 145, in create_dataloaders
    df, label_names, split_to_locs = load_dataset_csv(
  File "/home/jdj/work/ghoti-2021/CameraTraps/classification/train_utils.py", line 218, in load_dataset_csv
    with open(label_index_json_path, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'run_ghoti/label_index.json'
```

label_index.json은...
클래스 이름을 담은 JSON 파일인 것으로 보인다.
~/work/ghoti-2021/mydataset/cropped_coco_style/classlist.txt 를 복사하여
코드를 대충 참고해서 원하는 형태로 만들었다.

```
with open(label_index_json_path, 'r') as f:
  idx_to_label = json.load(f)
label_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
label_to_idx = {label: idx for idx, label in enumerate(label_names)}
```

이런 코드를 보니

```
{
    "0": "gm_f",
    "1": "lf_m",
    "2": "mv_f",
    "3": "pe_m",
    "4": "pf_f",
    "5": "pg_f",
    "6": "tg_f",
    "7": "tg_m",
    "8": "tm_f",
    "9": "tm_m",
    "10": "toc_f",
    "11": "toc_m"
}
```

label_index.json 은 이렇게 생겨야 할것이라고 유추가 가능했다.
내공이 많이 필요한 작업임.

```
    df['label_index'] = df['label'].map(label_to_idx.__getitem__)
  File "/home/jdj/usr/venv-3.9/lib/python3.9/site-packages/pandas/core/series.py", line 4160, in map
    new_values = super()._map_values(arg, na_action=na_action)
  File "/home/jdj/usr/venv-3.9/lib/python3.9/site-packages/pandas/core/base.py", line 870, in _map_values
    new_values = map_f(values, mapper)
  File "pandas/_libs/lib.pyx", line 2859, in pandas._libs.lib.map_infer
KeyError: 'ghoti'
```

돌아가는 꼴을 보아하니 JSON으로 만들었던 것을 csv로 만들어야 할 것 같다.
그렇게 판단하는 이유는?
csv의 label column을 label_index.json 에서 읽은 정보를 바탕으로 숫자로 변환하는 코드에서 에러가 났기 때문인데,
그 내용물이 ghoti라고함은 우리가 아까 만들었던 classification_ds.csv 파일 안에 넣은 내용이기 때문이다.

run_ghoti/classification_ds.csv 내용을 다시 보면.
label마다 dataset이 있는데...
그러면 아까 만든 dataset은 무용지물이고,
dataset이란게 뭔지도 모르겠다 이제.
코드를 봐도 COCO인지 뭔지 나오지도 않는다.

https://github.com/microsoft/CameraTraps/issues/259

결국 못해먹겠다고 issue를 남겼다. 그리고 댓글도 아주 빨리 받을 수 있었다.

```
Great question, and sorry for the confusion. I've updated the text on the main page to clarify what I'm about to say here...

We don't have a ready-to-go pipeline for classifier training, as the vast majority of our work is focused on MegaDetector. There are two frameworks for training classifiers in the "classification" folder, both likely-less-than-helpful-for-you for different reasons:

The main classification framework (described in the main README) is up to date, but depends on an internal database of labeled images (referred to in the documentation as "MegaDB").
The tutorial you referred to is completely standalone, but uses what are now obsolete dependencies.
If you are looking to train species classifiers, here are a few general paths you might go down, depending on your scenario:

If you want to use MegaDetector to generate crops, and train your classifier just on cropped animals, you can...
Run MegaDetector, use crop_detections.py to extract crops as individual images, then use whatever your favorite ML framework/tutorial is to train a classifier.
Use our current classification framework, but modify it to read labels from somewhere other than our internal database (which would still allow you to use, e.g., our inference and scoring tools). We are unlikely to be able to take this on as a project, but would welcome contributions that decouple our classification framework from MegaDB.
Modernize the older tutorial for standalone classifier training. Again, we would welcome this as a contribution to the repo.
If you want to train a classifier on whole images (rather than crops), you probably won't derive much benefit from the tools we have here.
Whichever way you proceed, one thing you may find useful from our repo is the MegaDetector batch API output format, which has a defined schema for classification results. Some OSS camera trap image review tools - for example, Timelapse - know how to read this format; this is how users interact with results of the classifiers we train.

Hope that helps, and thanks for reminder to clarify this outdated information on the main page.

Thanks!

-Dan
```

결국 out of date인 것이 맞고 we welcome your contribution이라고 한다.

## Starting From Scratch

PyTorch tutorial에서 가장 유사한 예제에서 출발하는 것을 추천한다.
한편 PyTorch Lightning framework에서 작업하는 것이 코딩해야 할 양이 줄어들어서 더욱 추천된다.

### Define Lightning Module

https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html

```
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

이걸 보자.
튜토리얼대로 하면 금방 만들 수 있는 편이다.
게다가 이전에 COCO style dataset으로 다행히도 만들어두었고, 이를 활용 가능하다.

```
dataset = torchvision.datasets.CocoDetection(
    root="mydataset/cropped_coco_style",
    annFile="mydataset/cropped_coco_style/train.json",
    transform=torchvision.transforms.ToTensor()
)
```

이렇게.

```
    net = MyResnet()

    train_dataset = torchvision.datasets.CocoDetection(
        root="mydataset/cropped_coco_style",
        annFile="mydataset/cropped_coco_style/train.json",
        transform=torchvision.transforms.ToTensor()
    )

    # for x, y in dataset:
    #     print(x.shape)
    #     print(y)
    # print(len(dataset))
    # x: torch.Size([3, 440, 1320])
    # y: [{'id': 639, 'image_id': 639, 'category_id': 7}]

    train_loader = DataLoader(train_dataset)
    trainer = pl.Trainer()
    trainer.fit(net, train_loader)
```

실행해보면

```
  File "/home/jdj/work/ghoti-2021/ghoti/model.py", line 9, in __init__
    self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
```

여기서가 이런 에러가 나오면서 문제가 발생한다.
쉬운 게 없다.
torch.hub.load + 에러메시지로 구글링 해보면 아직 진행중인 버그이며 (https://github.com/pytorch/vision/issues/4156)
walk-around는

```
model = torchvision.models.densenet121(pretrained=True, progress=True)
```

이런 식으로 모델을 불러오라는 것이다.
우린

```
self.resnet = torchvision.models.resnet50(pretrained=True, progress=True)
```

이렇게 변경했다.
