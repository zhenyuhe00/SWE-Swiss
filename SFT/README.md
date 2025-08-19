# Supervised Fine-Tuning (SFT) Instructions
This document provides instructions for fine-tuning the model. While we utilize our internal infrastructure for training (typically with a dynamic batch size of around 60), you can achieve similar results using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) library.
Follow these steps to set up the environment and run the fine-tuning process.

### 1. Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip3 install -e ".[torch,metrics]" --user
cd ..
```

### 2. Install Additional Dependencies

```bash
pip3 install deepspeed==0.16.0 --user
pip3 install transformers==4.51.3 --user
pip3 install pyarrow==20.0.0 --user
pip3 install git+https://github.com/linkedin/Liger-Kernel.git
```

### 3. Configure the Dataset
`LLaMA-Factory` uses a `dataset_info.json` file to manage and locate datasets. Copy the provided `dataset_info.json` from this repository into the `LLaMA-Factory` data directory.

```bash
cp dataset_info.json LLaMA-Factory/data/
```

### 4. Launch the Training Script
The following command initiates the distributed training process across multiple nodes.
Note: This command is configured for a multi-node environment. You may need to adjust `NNODES`, `MASTER_ADDR`, `NODE_RANK`, and other parameters based on your specific hardware and Slurm/cluster configuration.
```bash
cd LLaMA-Factory
# Set environment variables and launch the training script
# IMPORTANT: Replace $MASTER_ADDR and $NODE_RANK with your cluster's environment variables.
# In our experiments, we use 64 A100 GPUs for training.
DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR NODE_RANK=$NODE_RANK MASTER_PORT=10086 llamafactory-cli train ../train_qwen32b.yaml
```