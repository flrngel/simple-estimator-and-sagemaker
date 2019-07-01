from sagemaker.tensorflow.serving import Model
model = Model(model_data='s3://<AWS_BUCKET>/<S3_MODEL_PATH>',
              framework_version='1.13',
              role='<AWS_ROLE>')
predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')
