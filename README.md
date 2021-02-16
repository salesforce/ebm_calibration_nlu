


# Joint Energy-based Model Training for Better Calibrated Natural Language Understanding Models

## Introduction

This repo contains code for "Joint Energy-based Model Training for Better Calibrated Natural Language Understanding Models".

Our paper will appear in EACL 2021.

Paper link: https://arxiv.org/pdf/2101.06829.pdf

## Installation

We use the following packages, we recommend you to use a virtual environment (e.g. conda).
- Python 3.7
- torch==1.5.0
- transformers==2.11.0
  
Also, please run:

```pip install -r requirements.txt```

Finally, we assume that GPU (CUDA) is available .

## Usage

### Data Preparation

for glue data, go to ebm_calibration_nlu/data-sets/glue, and run
```python download_glue_data.py```

### Baseline Training

To reproduce the Roberta-base baseline, please run

```TASK_NAME=QNLI ./scripts/glue_baseline.sh train```

Note that in the example commands, I will use QNLI as an example. For other tasks, just switch from QNLI to SST-2 / QQP/ MNLI , etc.
After training, you can use the "eval" command to only get test-set result.

### Noise LM Model Training

As explained in the paper, before conducting NCE training, we need to first prepare noise data.

First, let's train a noise LM with the MLM objective.

``` MODE=partial2full RATE=0.4 TASK_NAME=QNLI bash ./scripts/lm_ncenoise.sh train ```

For QNLI, you should get a test-PPL around 3.64 .
Next, to generate noise samples:

``` MODE=partial2full RATE=0.4 TASK_NAME=QNLI bash ./scripts/lm_ncenoise.sh gen_save ```

### Energy Model Training

With the noise samples generated, we can now conduct NCE training. 

To run the "scalar" variant:

``` LMMODE=partial2full LMP2FRATE=0.4 NMODE=normal NRATIO=8 TASK_NAME=QNLI ./scripts/glue_nce.sh train ```

To run the "hidden" variant:

``` LMMODE=partial2full LMP2FRATE=0.4 NMODE=hidden NRATIO=8 TASK_NAME=QNLI ./scripts/glue_nce.sh train ```

To run the "sharp-hidden" variant:

``` LMMODE=partial2full LMP2FRATE=0.4 NMODE=selflabeled NRATIO=8 TASK_NAME=QNLI ./scripts/glue_nce.sh train ```

Again, after training you can use the "eval" command to just re-run the testing.
You will get the accuracy / ece number at the end of the log.


## Citation


``` 
@misc{he2021joint, title={Joint Energy-based Model Training for Better
Calibrated Natural Language Understanding Models}, author={Tianxing He and
Bryan McCann and Caiming Xiong and Ehsan Hosseini-Asl}, year={2021},
eprint={2101.06829}, archivePrefix={arXiv}, primaryClass={cs.CL} } 
```


## License

Please see [LICENSE](https://github.com/salesforce/ebm_calibration_nlu/blob/main/LICENSE.md).


