import sagemaker
import os
from sagemaker import get_execution_role
from sagemaker import image_uris
from sagemaker.tensorflow import TensorFlow
from ipdb import set_trace

sagemaker_session = sagemaker.Session()

role = "arn:aws:iam::903025558691:role/shibox-sagemaker-role"
region = sagemaker_session.boto_region_name
_instance = "ml.c5.4xlarge"
image_url = image_uris.retrieve(
    framework="tensorflow",
    region="us-west-2",
    version="2.8.0",
    image_scope="training",
    instance_type=_instance,
)

estimator = TensorFlow(
    entry_point="./word_embeddings/wd_em.py",
    image=image_url,
    role=role,
    instance_count=1,
    instance_type=_instance,
    framework_version="2.8",
    py_version="py39",
    checkpoint_s3_uri="s3://shibox-general/Models",
    # checkpoint_local_path="/"
)

train_dir = "s3://shibox-general" + "/aclImdb"
estimator.fit(train_dir)
