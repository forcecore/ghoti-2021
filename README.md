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

2_preproc.sh 를 작성했다. `bash 2_preproc.sh`로 실행하면 된다.

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
3_make_json.py 를 실행하면 된다.
