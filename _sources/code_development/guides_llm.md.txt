
# Running LLM Models in a Cluster Environment
This guide provides a comprehensive walkthrough on how to deploy and run Large Language Models (LLMs) within a cluster environment(Saga, Betzy, or Fram). With the growing availability of open-source LLMs, such as those hosted on [Hugging Face](https://huggingface.co/), leveraging these models has never been easier.

In this tutorial, we’ll focus on the [facebook/seamless-m4t-large](https://huggingface.co/facebook/seamless-m4t-large) model.  SeamlessM4T is a powerful collection of models designed to deliver high-quality translations for both speech and text, making it an excellent choice for multilingual and multimodal applications.



## Running LLM Models in a Virtual Environment

Running LLM models in a virtual environment is a straightforward and efficient method for deploying the program directly on the cluster. By using a virtual environment, you can isolate dependencies and install all the necessary software without affecting the system-wide configuration.

### Getting Started
In this section, we’ll write a simple Python script to load an audio file and use the LLM to generate a high-quality translation. Follow the steps below to set up your environment and organize your project:

**Step 1:  Create a Data Folder**

Begin by creating a folder named data inside your project directory. This folder will serve as a centralized location to store your input files.

**Step 2: Add a Sample Audio File**

Place a sample audio file in the data folder. For best results, use an audio file in .wav format. This ensures compatibility with the code we’ll write later and simplifies the process of loading the file for translation.

````bash
project-directory/
├── data/
│   └── sample_audio.wav
````

By organizing your files in this way, you’ll streamline the process of running the LLM model in a virtual environment and ensure that your project remains clean and manageable.


### Sample python code to perform speech to text translation
````python
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
import torchaudio
import torch
import os
print(torchaudio.get_audio_backend())
torchaudio.set_audio_backend("soundfile")
class FacebookSeamless:
    def __init__(self):
        print("Initializing the processor and model...")
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Model loaded and moved to device:", self.device)

    def transcribe_audio(self, file_path):
        try:
            print("Attempting to load audio from:", file_path)
            audio, orig_freq = torchaudio.load(file_path)
            print("Audio loaded successfully. Original frequency:", orig_freq)
            
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
                print("Audio was multi-channel, converted to mono.")
            
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
            print("Audio resampled to 16000 Hz.")
            
            audio = audio / torch.max(torch.abs(audio))
            print("Audio normalized.")
            
            audio_inputs = self.processor(audios=audio, sampling_rate=16000, return_tensors="pt").to(self.device)
            print("Audio inputs processed and moved to device.")
            
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang="eng",
                max_length=100,
                num_beams=3,
                length_penalty=1.0,
                early_stopping=True
            )
            print("Output tokens generated.")
            
            translated_text_from_audio = self.processor.batch_decode(output_tokens, skip_special_tokens=True)
            full_text = translated_text_from_audio[0]
            print("Translation completed.")
            return full_text
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return []

def main(audio_filename):
    print("Starting transcription process...")
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    audio_file_path = os.path.join(data_folder, audio_filename)
    print("Constructed file path:", audio_file_path)
    
    transcriber = FacebookSeamless()
    transcription = transcriber.transcribe_audio(audio_file_path)
    print("Transcription:", transcription)

if __name__ == "__main__":
    audio_filename = 'harvard.wav'
    main(audio_filename)
````



### Preparing the Environment

Before running the script, you need to set up your environment. This tutorial assumes that you are familiar with transferring project files from your local machine to the cluster. If you need assistance with file transfers, refer to the [File Transfer](../getting_started/file_transfer.md). 

Once the necessary files are in your working directory, you can proceed to create a virtual environment to install the additional packages required for running the Facebook Seamless model.

#### Loading the Python Module

To create a virtual environment, you first need to load the appropriate Python module available on the cluster. Note that the Python module version may vary over time, so it’s a good practice to check for the latest version before proceeding. Use the following command to list the available Python modules:

`module avail python` 

Once you’ve identified the appropriate version, load the module. For example, to load Python version `3.12.3` with `GCCcore-13.3.0`, use the following command:

``` module load Python/3.12.3-GCCcore-13.3.0```

This step ensures that you have the necessary Python environment to create and manage your virtual environment.

#### Creating and Activating the Virtual Environment
After loading the appropriate Python module, the next step is to create a virtual environment. This allows you to isolate dependencies and manage packages specific to your project. Use the following command to create the virtual environment:
 
 ```python -m venv myenv```
 
Once the virtual environment is created, activate it with the following command:
 
 ```source myenv/bin/activate```

When activated, your terminal prompt will change to indicate that you are now working within the virtual environment.

#### Installing Required Packages
With the virtual environment activated, you can now install the necessary packages for running the Seamless model. Below is the list of essential packages you’ll need to install:


```bash
pip install transformers
pip install git+https://github.com/huggingface/transformers.git sentencepiece
pip install torchaudio
pip install soundfile
pip install protobuf

```
After installing these packages, you can deactivate the virtual environment using this command:

```deactivate```

### List of packages that were used to run the program 
```bash
certifi                  2025.4.26
cffi                     1.17.1
charset-normalizer       3.4.2
filelock                 3.18.0
fsspec                   2025.3.2
huggingface-hub          0.31.2
idna                     3.10
Jinja2                   3.1.6
MarkupSafe               3.0.2
mpmath                   1.3.0
networkx                 3.4.2
numpy                    2.2.5
nvidia-cublas-cu12       12.6.4.1
nvidia-cuda-cupti-cu12   12.6.80
nvidia-cuda-nvrtc-cu12   12.6.77
nvidia-cuda-runtime-cu12 12.6.77
nvidia-cudnn-cu12        9.5.1.17
nvidia-cufft-cu12        11.3.0.4
nvidia-cufile-cu12       1.11.1.6
nvidia-curand-cu12       10.3.7.77
nvidia-cusolver-cu12     11.7.1.2
nvidia-cusparse-cu12     12.5.4.2
nvidia-cusparselt-cu12   0.6.3
nvidia-nccl-cu12         2.26.2
nvidia-nvjitlink-cu12    12.6.85
nvidia-nvtx-cu12         12.6.77
packaging                25.0
pip                      24.0
protobuf                 6.31.0
pycparser                2.22
PyYAML                   6.0.2
regex                    2024.11.6
requests                 2.32.3
safetensors              0.5.3
sentencepiece            0.2.0
setuptools               80.7.1
soundfile                0.13.1
sympy                    1.14.0
tokenizers               0.21.1
torch                    2.7.0
torchaudio               2.7.0
tqdm                     4.67.1
transformers             4.52.0.dev0
triton                   3.3.0
typing_extensions        4.13.2
urllib3                  2.4.0
```

### Writing the Job Script
To run your Hugging Face model for speech-to-text translation on the cluster, you’ll need to create a SLURM job script. This script will define the resources required for the job and execute the necessary commands. Below is an example of a SLURM job script:


```bash
#!/bin/bash -l
#SBATCH --job-name=llmTest_job
#Project account
#SBATCH --account=nnxxxx
#SBATCH --output=output.log
#SBATCH --error=error.log

#SBATCH --time=0:30:00
#SBATCH --partition=accel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet reset  # Reset the modules to the system default
module load Python/3.12.3-GCCcore-13.3.0  # Load the Python module
module list  # List loaded modules for debugging


#Activate virtual environment
source myenv/bin/activate


# Start GPU monitoring in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_usage.csv & 
GPU_MONITOR_PID=$!


# Run the script
python script.py

#Stop GPU monitoring
kill $GPU_MONITOR_PID

```


### Explanation of the Job Script

#### Job Name and Account:
- `#SBATCH --job-name=llmTest_job`: Specifies the name of the job.
- `#SBATCH --account=nnxxxx`: Specifies the project account to be charged for the job.

#### Output and Error Files:
- `#SBATCH --output=output.log`: Specifies the file where the standard output (stdout) will be saved.
- `#SBATCH --error=error.log`: Specifies the file where the standard error (stderr) will be saved.

#### Job Time and Partition:
- `#SBATCH --time=0:30:00`: Defines the maximum runtime for the job (30 minutes in this case).
- `#SBATCH --partition=accel`: Specifies the partition to use, which is `accel` in this case.

#### Resource Allocation:
- `#SBATCH --ntasks=1`: Specifies the number of tasks (1 task in this case).
- `#SBATCH --cpus-per-task=1`: Specifies the number of CPUs per task (1 CPU in this case).
- `#SBATCH --mem=10G`: Specifies the amount of memory allocated to the job (10 GB in this case).
- `#SBATCH --gpus=1`: Allocates 1 GPU for the job.

#### Job Environment Setup:
- `set -o errexit`: Exits the script if any command fails.
- `set -o nounset`: Treats unset variables as an error.
- `module --quiet reset`: Resets the modules to the system default.
- `module load python`: Loads the necessary Python module.
- `module list`: Lists the loaded modules for debugging purposes.

#### Activate Virtual Environment:
- `source myenv/bin/activate`: Activates the virtual environment.

#### GPU Monitoring:
- `nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_usage.csv &`: Starts GPU monitoring in the background and logs the usage to `gpu_usage.csv`.
- `GPU_MONITOR_PID=$!`: Captures the process ID of the GPU monitoring command.

#### Running the Program:
- ` python script.py`: Runs the `script.py` script ensuring that the  virtual environment activated.

#### Stopping GPU Monitoring:
- `kill $GPU_MONITOR_PID`: Stops the GPU monitoring process.



### Capturing the Output
When running the Seamless model, you can capture the output to verify that the audio file has been successfully translated using the LLM model. Below is an example of the output generated during the translation process:


```bash
Starting transcription process...
Constructed file path: /cluster/work/users/your_username/llmTest/data/harvard.wav
Initializing the processor and model...
Model loaded and moved to device: cuda
Attempting to load audio from: /cluster/work/users/your_username/llmTest/data/harvard.wav
Audio loaded successfully. Original frequency: 44100
Audio was multi-channel, converted to mono.
Audio resampled to 16000 Hz.
Audio normalized.
Audio inputs processed and moved to device.
Output tokens generated.
Translation completed.
Transcription: the stale smell of old beer lingers it takes heat to bring out the odour a cold dip restores health and zest a salt pickle tastes fine with ham tacos al pastor are my favorite a zestful food is the hot cross bun
```

You can also monitor GPU utilization by logging it to the output file. This can be achieved by including the appropriate command in your SLURM job script. Once the job completes, the GPU utilization details will be available in the output file. An example of the logged GPU utilization is shown below:

```bash
timestamp, utilization.gpu [%], utilization.memory [%]
2025/05/16 21:55:33.264, 4 %, 0 %
2025/05/16 21:55:34.264, 3 %, 0 %
2025/05/16 21:55:35.265, 29 %, 1 %
2025/05/16 21:55:36.265, 29 %, 1 %

```

### Conclusion
By following this guide, you can successfully run any Large Language Model (LLM) from Hugging Face on the cluster. This process allows you to leverage the power of open-source LLMs for tasks such as speech-to-text translation, all while utilizing the computational resources of the cluster efficiently.
