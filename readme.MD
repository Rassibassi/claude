# Claude
This repository provides helper functions and models for end-to-end learning of optical communication systems.

In particular, this repository provides an implementation for our paper "Deep Learning of Geometric Constellation Shaping including Fiber Nonlinearities" [https://arxiv.org/abs/1805.03785] and our manuscript "Geometric Constellation Shaping for Fiber Optic Communication Systems via End-to-end Learning" [https://arxiv.org/abs/1810.00774].

### Content
- Fiber model [iPyton Notebook Example](examples/np_NLINmodel.ipynb)
- End-to-end learning:

|iPyton Notebook Example | Channel | Optimised for bits/symbols (GMI/MI) | Gradient Through Channel | Reference |
|---|---|---|---|---|
[Link](examples/tf_AutoEncoderForGeometricShapingAndAwgn.ipynb) | AWGN | Symbols (MI) | :heavy_check_mark: | ---
[Link](examples/tf_AutoEncoderForGeometricShapingAndNlinModel.ipynb) | Fiber (NLIN/GN)	| Symbols (MI) | :heavy_check_mark: | [Arxiv 1](https://arxiv.org/abs/1805.03785), [Arxiv 2](https://arxiv.org/abs/1810.00774)
[Link](examples/tf_AutoEncoderForGeometricShapingAwgnGmi.ipynb) | AWGN | Bits (GMI) | :heavy_check_mark: | ---
[Link](examples/tf_AutoEncoderForGeometricShapingNlinGmi.ipynb) | Fiber (NLIN/GN)	| Bits (GMI) | :heavy_check_mark: | ---
[Link](examples/tf_AutoEncoderForGeometricShapingAndBlindAwgn.ipynb) | AWGN | Symbols (MI) | :x: | [Arxiv](https://arxiv.org/abs/1804.02276)

### Acknowledgements
The learning algorithm learns by embedding a fiber channel model within an autoencoder. The implementation of the fiber channel model is copied from a MATLAB implementation by Dar et al. [https://www.osapublishing.org/oe/abstract.cfm?uri=oe-22-12-14199, https://arxiv.org/abs/1310.6137]. Please cite the authors appropriately if you use the here provided Python implementation.

[Numpy model](claude/claude/models/NLIN.py) [claude/claude/models/NLIN.py]

[Tensorflow model](claude/claude/claudeflow/models/NLIN.py) [claude/claude/claudeflow/models/NLIN.py]

### Installation using Anaconda on Ubuntu
Install Anaconda from https://www.anaconda.com/download/

Clone the git repository into your filesystem
```
git clone https://github.com/rassibassi/claude
```
Enter the git repository
```
cd claude
```
Create a new conda environment
```
conda create -n claudeDev python=3.6 pip scipy jupyter matplotlib
```
Activate the new environment
```
source activate claudeDev
```
Install TensorFlow as instructed here: https://www.tensorflow.org/install/install_linux#InstallingAnaconda

Install the local `claude` pip package
```
pip install -e claude
```
This will install `claude` from the claude folder.

Check if everything is installed
```
pip list | grep -E 'claude|tensorflow|numpy|scipy|jupyter|matplotlib'
```
Output:
```
claude             0.1       /home/path-to-some-folder/claude
jupyter            1.0.0     
jupyter-client     5.2.3     
jupyter-console    5.2.0     
jupyter-core       4.4.0     
matplotlib         2.2.3     
numpy              1.15.1    
scipy              1.1.0     
tensorflow         1.10.1
```

Start Jupyter with
```
jupyter notebook
```

Open the Jupyter notebooks in the example directory.
Then you can click on Kernel -> Restart & Run All




