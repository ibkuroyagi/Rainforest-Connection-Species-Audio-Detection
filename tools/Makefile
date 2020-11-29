PYTHON:= python3.6
CUDA_VERSION:= 10.0
PYTORCH_VERSION:= 1.4
DOT:= .
.PHONY: all clean

all: virtualenv

virtualenv:
	virtualenv -p $(PYTHON) venv
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cu$(subst $(DOT),,$(CUDA_VERSION))/torch_stable.html
	. venv/bin/activate; cd ../; pip install -r requirements.txt
	. venv/bin/activate; mkdir ../input/; cd ../input/; kaggle competitions download \
		-c rfcx-species-audio-detection
	cd ../input/; unzip rfcx-species-audio-detection.zip -d rfcx-species-audio-detection; \
		rm rfcx-species-audio-detection.zip

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
