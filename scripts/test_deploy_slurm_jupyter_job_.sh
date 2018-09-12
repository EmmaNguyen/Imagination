
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_notebook_%j.log
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --time=04:00:00
date;hostname;pwd
 
module add jupyter

launch_jupyter_notebook
