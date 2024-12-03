# CSCE-614 Term Project: Mad-Max

This project is inspired by the MAD-Max paper and focuses on design space exploration to enhance the performance of machine learning models through advanced parallelization strategies. Using the VTrain simulator, we implement intra-node parallelization with Tensor Parallelism (TP) and inter-node parallelization with Fully Sharded Data Parallelism (FSDP).Additionally, modifications are applied to the Transformer, Embedding, and Hidden Layers to optimize the architecture. Various parallelization strategies, including tensor, data, and pipeline parallelism, are explored to determine their effectiveness. The project evaluates model performance across various configurations and analyzes the impact of modifying parameters in the configuration file on the performance of Mad-Max.



## Project Hierarchy

Please focus your attention on the mad-max branch. The main branch is just a fork from the vTrain repository. The project directory is organized as follows:

- **apex/**: Contains Apex library files for mixed precision training and optimizations. (Optional)
- **config/**: Includes configuration files for various models with different layer numbers (e.g., 96 and 128), as well as modifications for tensor, data, and pipeline parallelism.
- **data/**: Contains measured datpoints for validation of vTrain
- **docker/**: Contains Dockerfile for creating reproducible environments for the simulator.
- **profiler/**: Holds vTrain profiler to collect all CUDA traces between init_trace() and finish_trace().
- **src/**: Includes Python scripts for running the configurations and models, producing graphs, and etc.
- **trace/**: Contains trace files generated during simulation runs for further analysis.
- **results/**: Includes output .txt files showing predicted iterations, computations of GPUs and communication. Also includes some graphs.

Additional files include:
- **example.py**: A script showcasing how to execute the configurations.
- **process_configs.py** and **process_configs_parallel.py**: Scripts to handle configuration files and execute parallelization strategies.
- **requirements.txt**: Specifies the dependencies needed to run the project.

# Setup

To set up and run the project, follow these steps:

1. Clone this repository
   ```bash
    git clone "https://github.com/CSCE-614-Dr-Kim-Fall-2024/Group2_Mad-max.git"
   ```

2. [Download](https://drive.google.com/file/d/1NXe5qG41la2uFVsxbTyP714fnfeSyE1l/view?usp=drive_link) the `.sif` file and upload it along with the vTrain project files to the HPRC (High-Performance Research Computing environment).

3. Initiate a compute node:
   ```bash
   srun --nodes=1 --ntasks-per-node=4 --mem=30G --gpus=1 --time=01:00:00 --pty bash -i
   ```

4. Set the temporary cache directory:
   ```bash
   cd $SCRATCH
   export SINGULARITY_CACHEDIR=$TMPDIR/.singularity
   module load WebProxy
   ```

5. Navigate to the folder where `vtrain.sif` exists, and run the following command:
   ```bash
   singularity exec --nv --bind $(pwd):/workspace/vTrain2 vtrain.sif /bin/bash
   ```

6. Run the code using this command:
   ```bash
   python example.py -c config/<config_file>.json
   ```

Replace `<config_file>` with the desired configuration file name to test different setups.



# Group Members 

David	Hung  
Terencio	Martinez  
Sarah	Sotelo  
Sharan	Sekhar

# References
Research Paper -  MAD-Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems  
Research Paper - vTrain: A Simulation Framework for Evaluating Cost-effective and Compute-optimal Large Language Model Training  
[vTrain Github](https://github.com/VIA-Research/vTrain.git)  
[Nvidia Apex](https://github.com/NVIDIA/apex.git)
