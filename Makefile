.PHONY: all flatten preprocess distortion split normalize augment mel \
        tune-rnn tune-cnn1d tune-crdnn tune-all \
        train-rnn train-cnn1d train-crdnn train-all \
        eval-rnn eval-cnn1d eval-crdnn eval-all

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

tune-cnn1d:
	python -m src.tune --model cnn1d

tune-crdnn:
	python -m src.tune --model crdnn

tune-all: tune-rnn tune-cnn1d tune-crdnn

train-rnn:
	python -m src.train --model rnn

train-cnn1d:
	python -m src.train --model cnn1d

train-crdnn:
	python -m src.train --model crdnn

train-all: train-rnn train-cnn1d train-crdnn

eval-rnn:
	python -m src.evaluate --model rnn

eval-cnn1d:
	python -m src.evaluate --model cnn1d

eval-crdnn:
	python -m src.evaluate --model crdnn

eval-all: eval-rnn eval-cnn1d eval-crdnn
