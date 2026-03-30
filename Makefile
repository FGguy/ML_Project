.PHONY: all flatten preprocess distortion split normalize

all: flatten preprocess distortion split normalize

flatten:
	python -m src.data.flatten

preprocess:
	python -m src.data.preprocess

distortion:
	python -m src.data.distortion

split:
	python -m src.data.split

normalize:
	python -m src.data.normalize
