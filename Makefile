.PHONY: all flatten preprocess distortion split normalize augment mel \
        tune-rnn tune-crdnn-audio tune-crdnn tune-all \
        train-rnn train-crdnn-audio train-crdnn train-all \
        eval-rnn eval-crdnn-audio eval-crdnn eval-all

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

tune-rnn:
	python -m src.tune --model rnn

tune-crdnn-audio:
	python -m src.tune --model crdnn_audio

tune-crdnn:
	python -m src.tune --model crdnn

tune-all: tune-rnn tune-crdnn-audio tune-crdnn

train-rnn:
	python -m src.train --model rnn

train-crdnn-audio:
	python -m src.train --model crdnn_audio

train-crdnn:
	python -m src.train --model crdnn

train-all: train-rnn train-crdnn-audio train-crdnn

eval-rnn:
	python -m src.evaluate --model rnn

eval-crdnn-audio:
	python -m src.evaluate --model crdnn_audio

eval-crdnn:
	python -m src.evaluate --model crdnn

eval-all: eval-rnn eval-crdnn-audio eval-crdnn
