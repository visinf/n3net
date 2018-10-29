# Neural Nearest Neighbors Networks (NIPS*2018)

Official implementation of the denoising (PyTorch) and correspondence classification (Tensorflow) N3NET, that will be published in our NIPS paper:

Tobias Plötz and Stefan Roth, **Neural Nearest Neighbors Networks**, Advances in Neural Information Processing Systems (NIPS), 2018

Contact: Tobias Plötz (tobias.ploetz@visinf.tu-darmstadt.de)

# Installation

The denoising code is tested with Python 3.6, PyTorch 0.4.1 and Cuda 8.0 but is likely to run with newer versions of PyTorch  and Cuda.

To install PyInn run

    ```
    pip install git+https://github.com/szagoruyko/pyinn.git@master
    ```

Further requirements can be installed with 

    ```
    pip install -r requirements.txt
    ```

**A note on inference with Tensor Comprehensions.** Running test time inference on large images might exhaust your GPU memory. For that case, our code tries to run inference with [Tensor Comprehensions](https://facebookresearch.github.io/TensorComprehensions/). Please follow their instructions to setup TC in your python environment. If you have not installed TC in your environment inference will fall back to a slower computation.

Please download the BSDS500, Urban100 and Set12 datasets by cd'ing into `datasets/` and using the scripts provided therein. If you want to train your own Poisson-Gaussian denoising model, please additionally download the DIV2k dataset and the Waterloo dataset.

For setting up the correspondence classification code, please clone the repository (https://github.com/vcg-uvic/learned-correspondence-release), follow their instructions to setup your environment and copy the files located in `src_correspondence/`. This can also conveniently be done using the script `src_correspondence/clone_CNNet.sh`.

# Running the denoising code

To run the following commands, please cd into `src_denoising`.

To train a new model run:
    
    ```
    python main.py <options>
    ```

To test your model run:

    ```
    python main.py --eval --eval_epoch <epoch> --evaldir <dir>
    ```

To test our pretrained networks run:

    ```
    python main.py --eval --eval_epoch 51 --evaldir pretrained_sigma<25|50|70>
    ```

**A note on pretrained models.**
The pretrained models will give slightly different results than in the paper on Set5 due to differences in the random seed. 
On the other, larger datasets, results are as in the paper. 
Furthermore, in the paper we used a strong learning rate decay. 
Training our model again with a slower decay yields better results and we will add new pretrained models soon.

# Running the correspondence classification code

To run the following commands, please cd into `src_correspondence/CNNet`.

To train a new model run:

    ```
    python main.py --run_mode=train --net_arch=nips_2018_nl <options>
    ```

We provide two pretrained models. One is trained on the Brown indoor dataset, the other is trained on the St. Peters outdoor dataset. To test our pretrained models run:

    ```
    python main.py --run_mode=test --net_arch=nips_2018_nl  --data_te=<"brown_bm_3_05"|"st_peters"|"reichstag"> --data_va=<"brown_bm_3_05"|"st_peters"|"reichstag">
    --log_dir="<pretrained_brown|pretrained_stpeters>"
    --test_log_dir=<dir>
    ```

# Using the Neural Nearest Neighbors Block for Your Project

The core of the PyTorch implementation is located in `src_denoising/models/non_local.py` which provides classes for neural nearest neighbors selection (`NeuralNearestNeighbors`), a domain agnostic N3Block (`N3AggregationBase`) and a N3Block tailored towards image data (`N3Aggregation2D`). The file `src_denoising/models/n3net.py` contains the N3Net module that uses the 2D N3Block as non-local processing layer.

The core of the Tensorflow implementation is located in `src_correspondence/non_local.py` which provides analogous functionality as above.


## Citation
```
@inproceedings{Ploetz:2018:NNN,
  title     = {Neural Nearest Neighbors Networks},
  author    = {Pl\"otz, Tobias and Roth, Stefan},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  year = {2018}
}
```