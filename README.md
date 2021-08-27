# DeepFVE Layer Experiments

*Author: Dimitri Korsch*

This repository contains the official source code to produce results reported in the paper

> **End-to-end Learning of Fisher Vector Encodings for Part Features in Fine-grained Recognition.**<br>
> Dimitri Korsch, Paul Bodesheim and Joachim Denzler.<br>
> DAGM German Conference on Pattern Recognition (DAGM-GCPR), 2021.

## Installation:

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create an environment:
```bash
conda create -n chainer7cu11 python~=3.8.0 matplotlib jupyter opencv
conda activate chainer7cu11
```
***Note:** If you create environment with another name, then you need to prepend `CONDA_ENV=<another_name>` in the execution of the experiments.* Example: `CONDA_ENV=my_env ./train.sh`.
3. Install CUDA / cuDNN and required libraries:
```bash
conda install -c conda-forge cudatoolkit~=11.0.0 cudatoolkit-dev~=11.0.0 cudnn~=8.0.0 nccl cutensor
pip install -r requirements.txt
```

4. Install FVE-Layer implementation:
```bash
git submodule init
git submodule update
cd fve_layer
make
```



## Running the experiments (FGVC):

### Start [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) to log your experiment results

*`docker` and `docker-compose` are required*

***Note:** Sacred is <u>optional</u>, but if it is not running, you need to append `--no_sacred` in the execution of the experiments*. Example : `./train.sh --no_sacred`.

2. Go to `fgvc/sacred`
2. Copy the config template `config.sh.template` to `config.sh` and edit missing values.
3. Start the containers:
```bash
./run.sh

Recreating sacred_mongodb ... done
Recreating sacred_omniboard ... done
```
4. Check the containers:
```bash
docker ps

CONTAINER ID   IMAGE                            COMMAND                  CREATED          STATUS          PORTS                                                      NAMES
xxxxxxxxxxxx   vivekratnavel/omniboard:latest   "/sbin/tini -- yarn …"   38 seconds ago   Up 35 seconds   0.0.0.0:9000->9000/tcp, :::9000->9000/tcp                  sacred_omniboard
xxxxxxxxxxxx   mongo                            "docker-entrypoint.s…"   42 seconds ago   Up 39 seconds   27018/tcp, 0.0.0.0:27018->27017/tcp, :::27018->27017/tcp   sacred_mongodb

```
5. In your web browser go to [`localhost:9000`](http://localhost:9000) to open the omniboard.


### Download the model weights and datasets you want to test

1. Download the weights from **TODO: ADD LINKS TO THE MODELS** and put them in you data directory (e.g.: `<data_dir>/models/inception/model.imagenet.npz`)
2. Setup your datasets, so that they are located under `<data_dir>/datasets/<dataset_name>/ORIGINAL`.
3. Copy part annotations from **TODO: ADD LINKS TO THE PART ANNOTATIONS**. Link the contents of the `ORIGINAL` folder to `CS_parts` and copy the part annotations to `CS_PARTS/parts`.
4. Copy `example.yml` to `fgvc/info.yml` and update `BASE_DIR` to point to your data directory.

**Note:** An example data directory may look like the following:
```bash
data
├── datasets
│   ├── birds
│   │   ├── BIRDSNAP
│   │   │   ├── CS_parts
│   │   │   └── ORIGINAL
│   │   ├── CUB200
│   │   │   ├── CS_parts
│   │   │   │   ├── images -> ../ORIGINAL/images
│   │   │   │   ├── images.txt -> ../ORIGINAL/images.txt
│   │   │   │   ├── labels.txt -> ../ORIGINAL/labels.txt
│   │   │   │   ├── parts
│   │   │   │   │   ├── part_locs.txt
│   │   │   │   │   └── parts.txt
│   │   │   │   └── tr_ID.txt -> ../ORIGINAL/tr_ID.txt
│   │   │   └── ORIGINAL
│   │   │       ├── images
│   │   │       ├── images.txt
│   │   │       ├── labels.txt
│   │   │       ├── parts
│   │   │       │   ├── part_locs.txt
│   │   │       │   └── parts.txt
│   │   │       └── tr_ID.txt
│   │   └── NAB
│   │       ├── CS_parts
│   │       └── ORIGINAL
│   ├── dogs
│   │   ├── CS_parts
│   │   └── ORIGINAL
│   └── moths
│       ├── CS_parts
│       └── ORIGINAL
└── models
    └── inception
        ├── model.imagenet.ckpt.npz
        └── model.inat.ckpt.npz
```


### Run the training scripts
Change to the scripts directory and start the training script with default parameters:
```bash
cd fgvc/scripts
./train.sh
```

#### Some other examples of the training execution:

Prints the command instead of executing it. Useful to check the settings.
```bash
DRY_RUN=1 ./train.sh
python ../main.py train ../info.yml CUB200 GLOBAL --no_snapshot --n_jobs 3 --label_shift 1 --gpu 0 --model_type cvmodelz.InceptionV3 --prepare_type model --pre_training inat --input_size 299 --parts_input_size 299 --feature_aggregation concat --load_strict --fve_type no --n_components 1 --comp_size -1 --post_fve_size 0 --aux_lambda 0.5 --aux_lambda_rate 0.5 --aux_lambda_step 20 --ema_alpha 0.99 --init_mu 1 --init_sig 1 --mask_features --augmentations random_crop random_flip color_jitter --center_crop_on_val --batch_size 24 --update_size 64 --label_smoothing 0.1 --optimizer adam --epochs 60 --output .results/results/CUB200/adam/2021-08-27-10.45.53.421472597 --logfile .results/results/CUB200/adam/2021-08-27-10.45.53.421472597/output.log -lr 1e-3 -lrd 1e-1 -lrs 1000 -lrt 1e-8 --no_sacred

```
The most important parameters can be set by prepending variables to the training script (check config files under `fgvc/scripts/configs` for more information). This one starts NA-Birds training with CS-Parts on the second GPU in your system for the ResNet50 with a batch size of 16 without writing the results to sacred.
```bash
DATASET=NAB PARTS=CS_PARTS BATCH_SIZE=16 MODEL_TYPE=chainercv2.resnet50 GPU=1 ./train.sh --no_sacred
```

The selection of the **FVE implementation** can be controlled by the `FVE_TYPE` variable:

```bash
# training of CS-parts with GAP
PARTS=CS_parts FVE_TYPE=no ./train.sh

# ... with em-based FVE (our proposed method)
PARTS=CS_parts FVE_TYPE=em ./train.sh

# ... with gradient-based FVE (our implementation of Wieschollek et al.)
PARTS=CS_parts FVE_TYPE=grad ./train.sh
```
See `fgvc/scripts/config/21_fve.sh` config script for more configuration possibilities.

Further command line options of the training script can be seen here:
```bash
./train.sh -h

usage: main.py train [-h] --model_type
                     {chainercv2.resnet50,chainercv2.inceptionv3,cvmodelz.VGG16,cvmodelz.VGG19,cvmodelz.ResNet35,cvmodelz.ResNet50,cvmodelz.ResNet101,cvmodelz.ResNet152,cvmodelz.InceptionV3}
                     [--pre_training {imagenet,inat}]
                     [--input_size INPUT_SIZE [INPUT_SIZE ...]]
                     [--parts_input_size PARTS_INPUT_SIZE [PARTS_INPUT_SIZE ...]]
                     [--prepare_type {model,custom,tf,chainercv2}]
                     [--pooling {max,avg,tf_avg,g_avg,cbil,alpha}]
                     [--load LOAD] [--weights WEIGHTS] [--headless]
                     [--load_strict] [--load_path LOAD_PATH]
                     [--feature_aggregation {mean,concat}]
                     [--pred_comb {no,sum,linear}]
                     [--copy_mode {copy,share,init}]
                     [--label_shift LABEL_SHIFT] [--swap_channels]
                     [--n_jobs N_JOBS] [--shuffle_parts] [--logfile LOGFILE]
                     [--loglevel LOGLEVEL] [--gpu GPU [GPU ...]] [--profile]
                     [--only_klass ONLY_KLASS] [--fve_type {no,grad,em}]
                     [--n_components N_COMPONENTS] [--comp_size COMP_SIZE]
                     [--init_mu INIT_MU] [--init_sig INIT_SIG]
                     [--post_fve_size POST_FVE_SIZE] [--ema_alpha EMA_ALPHA]
                     [--aux_lambda AUX_LAMBDA]
                     [--aux_lambda_rate AUX_LAMBDA_RATE]
                     [--aux_lambda_step AUX_LAMBDA_STEP] [--mask_features]
                     [--no_gmm_update] [--only_mu_part] [--normalize]
                     [--no_sacred] [--augment_features] [--warm_up WARM_UP]
                     [--optimizer {sgd,rmsprop,adam}]
                     [--cosine_schedule COSINE_SCHEDULE] [--l1_loss]
                     [--from_scratch] [--label_smoothing LABEL_SMOOTHING]
                     [--only_head] [--seed SEED] [--batch_size BATCH_SIZE]
                     [--epochs EPOCHS] [--debug]
                     [--learning_rate LEARNING_RATE] [--lr_shift LR_SHIFT]
                     [--lr_decrease_rate LR_DECREASE_RATE]
                     [--lr_target LR_TARGET] [--decay DECAY]
                     [--augmentations [{random_crop,random_flip,random_rotation,center_crop,color_jitter} [{random_crop,random_flip,random_rotation,center_crop,color_jitter} ...]]]
                     [--center_crop_on_val]
                     [--brightness_jitter BRIGHTNESS_JITTER]
                     [--contrast_jitter CONTRAST_JITTER]
                     [--saturation_jitter SATURATION_JITTER] [--only_eval]
                     [--init_eval] [--no_progress] [--no_snapshot]
                     [--output OUTPUT] [--update_size UPDATE_SIZE]
                     [--test_fold_id TEST_FOLD_ID] [--analyze_features]
                     [--mpi] [--only_analyze] [--loss_scaling LOSS_SCALING]
                     [--opt_epsilon OPT_EPSILON]
                     data {CUB200,NAB,BIRDSNAP,DOGS,EU_MOTHS}
                     {GLOBAL,GT,GT2,CS_PARTS}
[...]
```
