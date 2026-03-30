.PHONY: all flatten preprocess distortion split normalize augment mel

all: flatten preprocess distortion split normalize augment mel

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

augment:
	python -m src.data.augment

mel:
	python -m src.data.mel_precompute
