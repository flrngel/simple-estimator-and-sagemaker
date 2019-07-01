from sagemaker.tensorflow.serving import Model
model = Model(model_data='s3://<AWS_BUCKET>/<S3_PATH_TO_MODEL>',
              framework_version='1.13',
              role='<AWS_ROLE>')
predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')
