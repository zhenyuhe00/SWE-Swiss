# Evaluation Results of SWE-Swiss-32B on SWE-bench Verified

This document outlines the evaluation pipeline and results for the **SWE-Swiss-32B** model on the **SWE-bench Verified** benchmark. Our methodology achieves a final score of **60.2%**.

## Evaluation Pipeline

Our evaluation process is adapted from the [Agentless framework](https://github.com/OpenAutoCoder/Agentless/tree/main). We have introduced the following three modifications to the standard pipeline:

1.  **Simplified Localization:** Inspired by [Agentless Mini](https://github.com/facebookresearch/swe-rl/tree/main), we streamline the localization stage. Instead of the original hierarchical process (file → class/function → final location), our model is only required to identify the relevant files for a given issue. We retain the use of an embedding-based model for this step as in the original Agentless approach.

2.  **Enriched Prompts for Patch Generation:** During the patch generation phase, we enhance the model's input prompt. In addition to the original issue description, we directly incorporate the full contents of the relevant files identified in the localization stage.

3.  **Enhanced Self-Consistency for Patch Selection:** For the final patch selection, we replace the standard self-consistency method (majority vote) with our own enhanced self-consistency algorithm. This allows for a more nuanced selection from the generated patches.

## Results

Our model achieved a final resolution score of **60.2%** on the 500 instances in the SWE-bench Verified dataset. 

### Result File Explanation

The final score is derived from the following result sets:

*   `40patches_results0`, `40patches_results1`, `40patches_results2`: These directories contain the outputs of three independent runs. In each run, we generated 40 candidate patches for every instance. A single patch was then selected based on unit test validation and our enhanced self-consistency method.
*   `120patches_results_merge012`: This directory contains the merged results from the three individual runs. The final score of 60.2% is calculated from this aggregated data.

### Reproducing the Score

You can reproduce the final score by running the following Python script.

```python
import json
import glob
from tqdm import tqdm
resolved_instances = []
instance_ids = set()
# Iterate through all result files in the merged directory
for result_dir in tqdm(glob.glob("120patches_results_merge012/*")):
    try:
        # Load the final score file
        score_file_path = f"{result_dir}/score_final.json"
        with open(score_file_path) as f:
            data = json.load(f)
        
        score = int(data['score'])
        instance_id = data['instance_id']
    except (FileNotFoundError, KeyError):
        score = 0
        instance_id = result_dir.split("/")[-1]
    # Ensure no duplicate instances are counted
    assert instance_id not in instance_ids, f"Duplicate instance ID found: {instance_id}"
    instance_ids.add(instance_id)
    resolved_instances.append(score)

total_instances = 500
final_score = (sum(resolved_instances) / total_instances) * 100
print(f"Score: {final_score:.1f}%")
```

## Complete Evaluation Traces
We provide the complete evaluation traces for full transparency and reproducibility:
* [40patches_results0](https://drive.google.com/file/d/1AVprnPGb0hchCwCb-iQRSR8midoMeAXP/view?usp=drive_link)
* [40patches_results1](https://drive.google.com/file/d/1p3TtRH0Ca49_iJhPeIV4e2d6hw6tCUAw/view?usp=sharing)
* [40patches_results2](https://drive.google.com/file/d/1y-3phaZ7ZlT2RkVoToJiPDrbqhf4QKiw/view?usp=sharing)
