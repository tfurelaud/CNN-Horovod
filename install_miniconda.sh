#!/bin/bash

cd

MINICONDA=Miniconda3-latest-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/$MINICONDA

bash $MINICONDA -b -p ~/anaconda
rm -f $MINICONDA
