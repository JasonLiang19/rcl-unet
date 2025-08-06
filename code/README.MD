


architecture.py - defines the ML. inherited from xxxxx.

custom_metrics.py - for handling variable sequence lengh. modified from xxxx.  


data_loading.py - contains functions to processes fasta sequencce file for encoding and scaling.

evaluate.py - to evaluate the effectiveness of different models.

serpin_classifier.py  - it uses a model built for identifying serpin. optional to run before predicting RCL.

train.py - for traininng the Unet model

utils.py - misc functions.



















 conda create -n rcl-env python=3.8.5
 conda activate rcl-env
 conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
 pip install -r requirements.txt
export LD_LIBRARY_PATH=/blue/zhou/jason.liang/conda/envs/rcl-env/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
