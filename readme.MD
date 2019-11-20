# Claude
This repository provides helper functions and models for end-to-end learning of optical communication systems.

For clarifications or bugs feel free to create a Github issue or write me an email: `chef@<firstname><lastname>.com`

This code has been developed as part of my PhD at DTU, Denmark, but also thereafter. There is still some code that has not been integrated into the public repository here, which will take some more time.

The [latest update](examples/tf_wdmSystem-learning.ipynb), an implementation of the split-step Fourier method (SSFM) in Tensorflow, allows gradient based learning end-to-end through the SSFM. The procedure is computational expensive and memory demanding, see here for two rendered iPython notebooks containing a pure [simulation](examples/tf_wdmSystem.html) and an example of [constellation shaping across the SSFM via end-to-end learning](examples/tf_wdmSystem-learning.html). Since the SSFM models cross-wavelength, cross-polarization and temporal correlations of the signal, in principle, it is possible to learn a high dimensional constellation shape across all mentioned degrees of freedom. However, the computational demand and memory usage is very large. If you happen to have a high performance cluster with many GPUs in your backyard, and are interested in attacking such high dimensional constellation shape, feel free to reach out. Also, if someone is willing to share her/his implementations (also in other programming languages) of fibre models modelling cross-wavelength, cross-polarization or temporal correlations on a symbol level, I'm happy to help with a Tensorflow implementation.

Further, this repository provides an implementation for two conference contributions ([https://arxiv.org/abs/1805.03785], [https://arxiv.org/abs/1907.08535]) and a manuscript [https://arxiv.org/abs/1810.00774].

### Content
- Fiber model, [NLIN/GN model](examples/np_NLINmodel.ipynb), [Split step Fourier Method](claude/claude/claudeflow/models/SSFstatic.py)
- System components, e.g. [Pulse shaping filter](examples/tf_pulseshaperB2B.ipynb), WDM system framework [back2back](examples/tf_wdmSystemB2B.ipynb) and [transmission simulation](examples/tf_wdmSystem.ipynb)
- Geometrical/Probabilistic constellation shaping via end-to-end learning:

|iPython Notebook Example | Shaping | Channel | Optimised for bits/symbols (GMI/MI) | Gradient Through Channel | Reference |
|---|---|---|---|---|---|
[Link](examples/tf_AutoEncoderForGeometricShapingAndAwgn.ipynb) | Geometric | AWGN | Symbols (MI) | :heavy_check_mark: | ---
[Link](examples/tf_AutoEncoderForProbabilisticShapingAndAwgn.ipynb) | Probabilistic | AWGN | Symbols (MI) | :heavy_check_mark: | [Arxiv](https://arxiv.org/abs/1906.07748)
[Link](examples/tf_AutoEncoderForJointGeoProbShapingAndAwgn.ipynb) | Joint Geo./Prob. | AWGN | Symbols (MI) | :heavy_check_mark: | [Arxiv](https://arxiv.org/abs/1906.07748)
[Link](examples/tf_AutoEncoderForGeometricShapingAndNlinModel.ipynb) | Geometric | Fiber (NLIN/GN)	| Symbols (MI) | :heavy_check_mark: | [Arxiv 1](https://arxiv.org/abs/1805.03785), [Arxiv 2](https://arxiv.org/abs/1810.00774)
[Link](examples/tf_AutoEncoderForProbabilisticShapingAndNlinModel.ipynb) | Probabilistic | Fiber (NLIN/GN) | Symbols (MI) | :heavy_check_mark: | ---
[Link](examples/tf_wdmSystem-learning.ipynb) | Geometric | Fiber (SSFM)| Symbols (MI) | :heavy_check_mark: | ---
[Link](examples/tf_AutoEncoderForGeometricShapingAwgnGmi.ipynb) | Geometric | AWGN | Bits (GMI) | :heavy_check_mark: | ---
[Link](examples/tf_AutoEncoderForGeometricShapingNlinGmi.ipynb) | Geometric | Fiber (NLIN/GN)	| Bits (GMI) | :heavy_check_mark: | [Arxiv](https://arxiv.org/abs/1907.08535)
[Link](examples/tf_AutoEncoderForGeometricShapingAndBlindAwgn.ipynb) | Geometric | AWGN | Symbols (MI) | :x: | [Arxiv](https://arxiv.org/abs/1804.02276)

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
tensorflow         1.14
```

Start Jupyter with
```
jupyter notebook
```

Open the Jupyter notebooks in the example directory.
Then you can click on Kernel -> Restart & Run All




