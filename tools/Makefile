PYTHON:= python3.6
CUDA_VERSION:= 10.1
PYTORCH_VERSION:= 1.6.0
DOT:= .
.PHONY: all clean

all: virtualenv

virtualenv:
	virtualenv -p $(PYTHON) venv
	. venv/bin/activate; pip install torch==1.6.0+cu$(subst $(DOT),,$(CUDA_VERSION)) \
		torchvision==0.7.0+cu$(subst $(DOT),,$(CUDA_VERSION)) -f https://download.pytorch.org/whl/torch_stable.html
	. venv/bin/activate; cd ../; pip install -r requirements.txt
	. venv/bin/activate; mkdir ../input/; cd ../input/; kaggle competitions download \
		-c rfcx-species-audio-detection
	cd ../input/; unzip rfcx-species-audio-detection.zip -d rfcx-species-audio-detection; \
		rm rfcx-species-audio-detection.zip
	. venv/bin/activate; mkdir ../input/; cd ../input/; kaggle datasets download \
		-d yasufuminakama/iterative-stratification
	cd ../input/; unzip iterative-stratification.zip; rm iterative-stratification.zip
		
apex:
	export CUDA_HOME=/usr/local/cuda-$(CUDA_VERSION)/
	git clone https://github.com/NVIDIA/apex.git
	. venv/bin/activate; cd apex; \
		pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
		# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

clean:
	rm -fr venv apex
	find -iname "*.pyc" -delete
