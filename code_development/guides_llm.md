
# Running LLM Models Inside the Cluster
This document provides an overview of how to run the Large Language Model (LLM) on any of the machines (Saga, Betzy, or Fram). There are numerous open-source LLM models available for download from [Hugging Face](https://huggingface.co/). This tutorial focuses on using one of the LLM models, [facebook/seamless-m4t-large](https://huggingface.co/facebook/seamless-m4t-large). SeamlessM4T is a collection of models designed to deliver high-quality translations from speech and text.


In this section, we will explore two methods for running our LLM model inside the cluster:

1. **Running LLM Models in a Virtual Environment**: This straightforward method involves running the program directly on the cluster.
2. **Running LLM levereging Containerized Solution**: This method involves running the container inside the cluster and requires prior knowledge of Docker and how to run Singularity images on the cluster.

We will first cover the virtual environment approach, followed by a detailed discussion of the containerized solution later in the documentation.

## Running LLM Models in a Virtual Environment

To get started, we will write a simple Python script that loads an audio file and uses the LLM to generate a high-quality translation from the audio file. Follow these steps:

1. **Create a Data Folder**: Inside your project directory, create a folder named `data`.
2. **Add Sample Audio File**: Place your sample audio file, preferably in `.wav` format, inside the `data` folder. This ensures that the code below can find the required audio file for the translation.

By organizing your files this way, you will streamline the process of running your LLM model in a virtual environment.


## Sample python code to perform speech to text translation
```python
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
    audio_filename = 'sample_irish.wav'
    main(audio_filename)
```

## Required packages to run the program inside the requirements.txt file
```bash
bokeh==3.4.1
cffi==1.17.0
contourpy==1.2.1
cycler==0.12.1
ffmpeg-python==0.2.0
flit_core==3.9.0
fonttools==4.53.0
future==1.0.0
Jinja2==3.1.4
kiwisolver==1.4.5
livelossplot==0.5.5
MarkupSafe==2.1.5
matplotlib==3.9.0
numpy==1.26.4
packaging==23.2
pandas==2.2.2
pillow==10.3.0
pip==24.0
protobuf==5.28.0
pycparser==2.22
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
setuptools==68.2.2
setuptools-scm==8.0.4
six==1.16.0
soundfile==0.12.1
tomli==2.0.1
tornado==6.4
typing_extensions==4.8.0
tzdata==2024.1
wheel==0.41.2
xyzservices==2024.4.0
```

## Setting Up the Environment

This tutorial expects that you are familiar with transferring the project into your working directory from the local machine.For more information on how to transfer the file see [File Transfer](../getting_started/file_transfer.md). Once you have these files inside the directory, you can create a virtual environment to load additional packages specific to running the Facebook Seamless model.

### Setting Up the Environment
First, load the module available in the cluster that allows you to create a virtual environment. The module you can load prior to creating the virtual environment is:


``` module load Python/3.8.6-GCCcore-10.2.0```

### Creating and Activating the Virtual Environment
 Once you load the Python module, you can create the virtual environment using the following command:
 
 ```python -m venv myenv```
 
 Activate the virtual environment with this command:
 
 ```source myenv/bin/activate```

 ### Installing Required Packages
For our Seamless model, there are few important packages that we need to install inside the virtual environment. They are listed below:


```bash
pip install transformers
pip install git+https://github.com/huggingface/transformers.git sentencepiece
pip install torchaudio

```
After installing these packages, you can deactivate the virtual environment:

```deactivate```

### Installing Required Packages
Next, create a SLURM script to run your Hugging Face model for performing speech-to-text translation. Below is an example SLURM job script:


```bash
#!/bin/bash -l
#SBATCH --job-name=llmTest_job
#Project account
#SBATCH --account=nnxxxx
#SBATCH --output=output_test.log
#SBATCH --error=error_test.log

#SBATCH --time=0:30:00
#SBATCH --partition=accel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module load Python/3.11.5-GCCcore-13.2.0  # Load the Python module
module list  # List loaded modules for debugging

#directory where the program is located
cd /cluster/work/users/your_user_name/llmTest

#create and  Activate virtual environment
python -m venv myenv
source myenv/bin/activate

#Install packages from requirements.txt
pip install -r requirements.txt

# Start GPU monitoring in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_usage.csv & 
GPU_MONITOR_PID=$!


# Run the script
python main.py

#Stop GPU monitoring
kill $GPU_MONITOR_PID

```
### Explanation of the Job Script
#### Job Name and Account:
- `#SBATCH --job-name=llmTest_job`: Specifies the name of the job.
- `#SBATCH --account=nnxxxx`: Specifies the project account to be charged for the job.
#### Output and Error Files:
- `#SBATCH --output=output_test.log`: Specifies the file where the standard output (stdout) will be saved.
- `#SBATCH --error=error_test.log`: Specifies the file where the standard error (stderr) will be saved.
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
- `module --quiet purge`: Resets the modules to the system default.
- `module load python`: Loads the necessary Python module.
- `module list`: Lists the loaded modules for debugging purposes.
#### Directory and Virtual Environment:
- `cd /cluster/work/users/your_user_name/llmTestDocker`: Changes the directory to where the program is located.
- `source myenv/bin/activate`: Activates the virtual environment.
#### GPU Monitoring:
- `nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_usage.csv &`: Starts GPU monitoring in the background and logs the usage to `gpu_usage.csv`.
- `GPU_MONITOR_PID=$!`: Captures the process ID of the GPU monitoring command.
#### Running the Program:
- ` python main.py`: Runs the `main.py` script ensuring that the  virtual environment activated.
#### Stopping GPU Monitoring:
- `kill $GPU_MONITOR_PID`: Stops the GPU monitoring process.
#### Deactivating the Virtual Environment:
- `deactivate`: Deactivates the virtual environment.
This job script sets up the environment, activates the virtual environment, monitors GPU usage, runs the main Python script inside the Singularity container, and ensures proper cleanup after the job completes.

In our case while running the seamless model we can capture the output as shown below where we can see our audio is translated using the LLM model.
```bash
Starting transcription process...
Constructed file path: /cluster/work/users/your_user_name/llmTest/data/sample_irish.wav
Initializing the processor and model...
Model loaded and moved to device: cuda
Attempting to load audio from: /cluster/work/users/your_user_name/llmTest/data/sample_irish.wav
Audio loaded successfully. Original frequency: 16000
Audio resampled to 16000 Hz.
Audio normalized.
Audio inputs processed and moved to device.
Output tokens generated.
Translation completed.
Transcription: Dogs, cats, pigs, cows, ant, antelope, dinosaur, leopard, lion, rat, ladybird, snake, spider, spider, mushroom, penguin, eagle, bluefinch, eagle, hair, hairy bear, cat, dog, lion, tiger, giraffe, hippopotamus, rhinoceros, whale, snail, catfish, salmon, trout, duck, deer, fly, dinosaur, dragonfly, and many other animals.
```

you could also visulize the GPU utilization from the output file where we log the GPU utilization by writing the command in the job script. In this case, the GPU utilization can be seen like this:
```bash
timestamp, utilization.gpu [%], utilization.memory [%]
2024/09/08 13:03:31.000, 100 %, 26 %
2024/09/08 13:03:32.000, 88 %, 22 %
2024/09/08 13:03:33.000, 0 %, 0 %
2024/09/08 13:03:34.001, 35 %, 19 %
2024/09/08 13:03:35.001, 35 %, 19 %
2024/09/08 13:03:36.001, 36 %, 19 %
2024/09/08 13:03:37.001, 38 %, 20 %
2024/09/08 13:03:38.002, 33 %, 17 %

```


### Running LLM levereging Containerized Solution
It is the recommended way to run your LLM models inside the cluster. This method ensures that all required packages are included in the `requirements.txt` file, which helps avoid dependency issues.
#### Steps:
1. **Create the Docker Image**: Include all necessary packages in the `requirements.txt` file and build the Docker image on your local machine.
2. **Transfer the Docker Image**: Transfer the Docker image to the cluster and place it in your work directory.
3. **Run the Singularity Image**: Pull the Docker image and run it as a Singularity image using a job script.
This approach ensures that your LLM runs as expected without any dependency errors. Additionally, you won't need to load any modules inside the cluster since all required packages are specified in the `requirements.txt` file.
#### Dockerfile and Requirements
Below are the contents of the `Dockerfile` and `requirements.txt` file for the same LLM project. Note that this method assumes the user has some familiarity with using Docker on their local machine.
#### Dockerfile
```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application code into the container
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main.py when the container launches
CMD ["python", "main.py"]
```

#### requirements.txt
```bash
bokeh==3.4.1
certifi==2024.7.4
cffi==1.17.0
charset-normalizer==3.3.2
contourpy==1.2.1
cycler==0.12.1
ffmpeg-python==0.2.0
filelock==3.15.4
flit_core==3.9.0
fonttools==4.53.0
fsspec==2024.6.1
future==1.0.0
huggingface-hub==0.24.6
idna==3.8
Jinja2==3.1.4
kiwisolver==1.4.5
livelossplot==0.5.5
MarkupSafe==2.1.5
matplotlib==3.9.0
mpmath==1.3.0
networkx==3.3
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.6.20
nvidia-nvtx-cu12==12.1.105
packaging==23.2
pandas==2.2.2
pillow==10.3.0
pip==24.0
protobuf==5.28.0
pycparser==2.22
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
regex==2024.7.24
requests==2.32.3
safetensors==0.4.4
sentencepiece==0.1.99
setuptools==68.2.2
setuptools-scm==8.0.4
six==1.16.0
soundfile==0.12.1
sympy==1.13.2
tokenizers==0.19.1
tomli==2.0.1
torch==2.4.0
torchaudio==2.4.0
torchvision==0.19.0
tornado==6.4
tqdm==4.66.5
triton==3.0.0
typing_extensions==4.8.0
tzdata==2024.1
urllib3==2.2.2
wheel==0.41.2
xyzservices==2024.4.0
git+https://github.com/huggingface/transformers.git
```

So, in the local machine inside your project directory where you have the main.py file which is given above and these two files you are ready to create your docker container. To create the docker container you can use the command given below:

```docker buildx build --platform linux/amd64 --no-cache -t llm-test:latest --load```

Note that, we are defining the platform as amd64 because in our case,that is the specific architecture of the  platform on the cluster where we would run our program.

Once the docker image is created you can check if the image is compatible with the architecture of your machine using this command:

```docker inspect llm-test:latest l grep Architecture```

The next step would be to save the created image in the .tar file so that you can transfer it to the cluster.Here is the command for that one.

```docker save llm-test:latest -o llm-test.tar```

Now, create a folder inside your working directory and give it a name. In our case, we named the directory `llmTestDocker` because we need to specify the path where we will send our project files.

To send your files to the cluster, use the following command. Note that you should be in the directory where you have the `.tar` file before executing this command:


```rsync llm-test.tar your_user_name@saga.sigma2.no:/cluster/work/users/your_user_name/llmtestDocker```


Now, after this, we will have the `.tar` file inside our cluster. We need to pull the Docker image. However, before doing that, we need to create a temporary file inside our user directory and set the environment path to that file. This is important because if we don't do this, pulling the Docker image will utilize a lot of cache space in the home directory, and we will easily exceed the disk quota space.

So, create the tmp file inside the user directory and then set the environment variable with this command given below:

```export SINGULARITY_TMPDIR=/cluster/work/users/your_user_name/tmp```

```export SINGULARITY_CACHEDIR=/cluster/work/users/your_user_name/tmp```

Once you set the path, you can go inside the directory where you have transferred your .tar file. Then, you can use this command to pull the docker image using singularity with the command given below:

```singularity pull docker-archive:///cluster/work/users/your_user_name/llmTestDocker/llm-test.tar```

Now, this will generate the singularity file for your docker image with .sif extension.
After this you will need to create the job script like before where you run your singularity image. Below is the job script that we used  in our case.


```bash
#!/bin/bash -l
#SBATCH --job-name=llmTest_job1
#SBATCH --account=nnxxxx
#SBATCH --output=output_test.log
#SBATCH --error=error_test.log
#SBATCH --time=0:30:00
#SBATCH --partition=accel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module list  # List loaded modules for debugging

# Directory where the program is located
cd /cluster/work/users/your_user_name/llmTestDocker

# Start GPU monitoring in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_usage.csv &
GPU_MONITOR_PID=$!

# Path to the Singularity image file
SINGULARITY_IMAGE=/cluster/work/users/your_user_name/llmTestDocker/llm-test.tar_latest.sif

# Run the script using Singularity with the necessary bind paths
singularity exec --nv $SINGULARITY_IMAGE python /app/main.py

# Stop GPU monitoring
kill $GPU_MONITOR_PID


```

Everything here is the same as the job script we created before. However, the major difference is that instead of loading the modules inside the cluster or setting up a virtual environment to load any other external modules that are not available in the cluster, we just run the Singularity image. This will run the program successfully.

Once you have this job script, you can simply run the program by submitting the Slurm job script with the following command:

```sbatch <job_script_name>.sh```

Then you can find the output like described previously.Following these approaches, you can run any LLM from the hugging face on the cluster.