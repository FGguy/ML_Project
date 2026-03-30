.PHONY: all flatten preprocess distortion split normalize augment

all: flatten preprocess distortion split normalize augment

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
