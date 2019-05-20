default:
	python app.py --learning_rate 1e-4 --epochs 1 --debug True --model_dir saved_model
clean:
	rm -rf saved_model
test:
	cd misc && python convert_image.py
	cd misc && saved_model_cli run --dir ../saved_model/*/ --tag_set serve --signature_def serving_default --inputs="images=sexy.npy"
