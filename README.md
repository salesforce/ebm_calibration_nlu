


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

We find that the official MRPC data has been changed recently, and it may disable our code to run.
Therefore, we provide an original copy of MRPC at data-sets/glue/MRPC_overwrite/ , you can overwrite data-sets/glue/glue_data/MRPC/ with it to enable our code to run.

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
We find that a mask rate of 0.4 works well for most tasks, however, for some task with fewer data, it's sometimes better to use a smaller rate. (Because the trained LM is not so good)
For MRPC / CoLA / RTE / WNLI, we tune the rate in {0.1, 0.2, 0.4} for each EBM variant (e.g., use RATE=0.2 in the command), and report the best result.

Next, to generate noise samples:

``` MODE=partial2full RATE=0.4 TASK_NAME=QNLI bash ./scripts/lm_ncenoise.sh gen_save ```

(Not necessary but recommended) For some large data-sets like QQP or MNLI, we need to generate more noise samples so that we won't use the same noise sample twice, please do this by:

``` SEED=2 MODE=partial2full RATE=0.4 TASK_NAME=MNLI bash ./scripts/lm_ncenoise.sh gen_save_seed ```
``` SEED=3 MODE=partial2full RATE=0.4 TASK_NAME=MNLI bash ./scripts/lm_ncenoise.sh gen_save_seed ```
``` SEED=4 MODE=partial2full RATE=0.4 TASK_NAME=MNLI bash ./scripts/lm_ncenoise.sh gen_save_seed ```
``` SEED=5 MODE=partial2full RATE=0.4 TASK_NAME=MNLI bash ./scripts/lm_ncenoise.sh gen_save_seed ```

You can run this in parallel as they won't write to the same file. These files will be automatically loaded by the data reader when we do NCE training.

### Energy Model Training

With the noise samples generated, we can now conduct NCE training. 

To run the "scalar" variant:

``` LMMODE=partial2full LMP2FRATE=0.4 NMODE=normal NRATIO=8 TASK_NAME=QNLI ./scripts/glue_nce.sh train ```

To run the "hidden" variant:

``` LMMODE=partial2full LMP2FRATE=0.4 NMODE=hidden NRATIO=8 TASK_NAME=QNLI ./scripts/glue_nce.sh train ```

To run the "sharp-hidden" variant:

``` LMMODE=partial2full LMP2FRATE=0.4 NMODE=selflabeled NRATIO=8 TASK_NAME=QNLI ./scripts/glue_nce.sh train ```

"NRATIO" refers to the noise ratio for the NCE objective. In our experiments, we tune it to be 1 or 8 (e.g., by setting NRATIO=1 in the command). In some cases, a small ratio of 1 actually works better than 8.

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

The code is released under the BSD-3 License - see [LICENSE](https://github.com/salesforce/ebm_calibration_nlu/blob/main/LICENSE.md) for details


