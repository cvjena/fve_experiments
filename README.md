# DeepFVE Layer Experiments

_Author: Dimitri Korsch_

Installation:

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create an environment:
```bash
conda create -n <myenv> python~=3.8.0 matplotlib jupyter opencv
conda activate <myenv>
```

3. Install CUDA / cuDNN and required libraries:
```bash
conda install -c conda-forge cudatoolkit~=11.0.0 cudnn nccl
pip install -r requirements.txt
```

4. _(optional)_ Install dependencies for tests
```bash
pip install -r requirements.dev.txt
```

