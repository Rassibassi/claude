# Claude
This repository provides helper functions and models for end-to-end learning of optical communication system.

In particular, this repository provides an implementation for our paper "Deep Learning of Geometric Constellation Shaping including Fiber Nonlinearities" [https://arxiv.org/abs/1805.03785].

### Content
[Example of fiber model](examples/np_NLINmodel.ipynb)

[Example of end-to-end learning of geometric constellation shapes](examples/tf_AutoEncoderForGeometricShapingAndNlinModel.ipynb)

### Acknowledgements
The learning algorithm learns by embedding a fiber channel model within an autoencoder. The implementation of the fiber channel model is copied from a MATLAB implementation by Dar et al. [https://www.osapublishing.org/oe/abstract.cfm?uri=oe-22-12-14199, https://arxiv.org/abs/1310.6137]. Please cite the authors appropriately if you use the here provided Python implementation.

[Numpy model](claude/claude/models/NLIN.py) [claude/claude/models/NLIN.py]

[Tensorflow model](claude/claude/claudeflow/models/NLIN.py) [claude/claude/claudeflow/models/NLIN.py]

### Installation using Anaconda on Ubuntu
Install Anaconda from https://www.tensorflow.org/install/install_linux#InstallingAnaconda

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
Install TensorFlow as instructed here: https://www.tensorflow.org/install/

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





