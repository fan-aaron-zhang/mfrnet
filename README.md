
### Instructions:

- Install Pycharm: there is a free version. Good for debugging.
- Install python 3.5 (I have 3.5.2) or 3.6 + libraries + Tensorflow gpu (I have 1.8.0)
- Python dependencies: Tensorlayer, numpy(1.16.1), scipy, matplotlib, sklearn and imageio 

### Usage


Run YUV model evaluation

--mode=evaluate
--evalHR=$output yuv path$
--evalLR=$input yuv path$
--testModel=$MFRNet_BVI_VTM70_SR_QPXX.npz$
--ratio=1
--nlayers=16
--GAN=0
--nframes=0
--eval_inputType=YUV
--readBatch_flag=1
--inputFormat=RGB

All input files should use the standard file name, e.g. Campfire_3840x2160_30fps_10bit_qp22.yuv (filename_HxW_xxxfps_xbit_qpxx.yuv)
### Reference

Please cite our papers if you use this code for your research

[1] Ma, Di, Fan Zhang, and David R. Bull. "MFRNet: a new CNN architecture for post-processing and in-loop filtering." IEEE Journal of Selected Topics in Signal Processing 15.2 (2020): 378-387.
