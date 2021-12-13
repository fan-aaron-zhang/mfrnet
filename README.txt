###################################################################################################
#### This code was developed by Mariana Afonso, Phd student @ University of Bristol, UK, 2021 #####
################################## All rights reserved © ##########################################

Instructions:

- Install Pycharm: there is a free version. Good for debugging.
- Install python 3.5 (I have 3.5.2) or 3.6 + libraries + Tensorflow gpu (I have 1.8.0)
- Python dependencies: Tensorlayer, numpy 1.16.1, scipy, matplotlib, sklearn and imageio 

Quick uses (examples):

- Run YUV model evaluation

--mode=evaluate
--evalHR=M:\Aaron\ViSTRA\TRAINING_RESULTS\HM1620_TRAINING_DB18_9bit\REC\QP27\Aamerican-football-scene4_3840x2160_60fps_10bit_420_qp21_BD09.yuv
--evalLR=M:\Aaron\ViSTRA\TRAINING_RESULTS\HM1620_TRAINING_DB18_9bit\REC\QP27\Aamerican-football-scene4_3840x2160_60fps_10bit_420_qp21_BD09.yuv
--testModel=0
--ratio=1
--nlayers=16
--GAN=0
--nframes=0
--eval_inputType=YUV
--readBatch_flag=1
--inputFormat=RGB