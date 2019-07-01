# simple-estimator-and-sagemaker 👋

> Simple guide to use tf.estimator and deploy to AWS SageMaker

## Usage

```sh
# train and test simple mnist model
$ make && make test
# upload your model
$ tar cvzf model.tar.gz saved_model/<MODEL_VERSION>/
$ aws cp model.tar.gz s3://<AWS_BUCKET>/<S3_MODEL_PATH>
# change AWS variables before run this
$ python sagemaker_deploy.py
$ python sagemaker_inference.py
```

## Author

- [@flrngel](https://github.com/flrngel)
- [@jun85664396](https://github.com/jun85664396)
