Machine learning outside CMSSW

Conda on hepgpu01:
```
source /local/HephyML/setup.sh
conda env create --file environment-tf16.yml
conda activate tf16
```

Conda on CBE:
Make a ~/.condarc file with:
```
envs_dirs:
  - /scratch-cbe/users/<username>/conda/envs
pkgs_dirs:
  - /scratch-cbe/users/<username>/conda/pkgs
```

"scratch-cbe/users" can be replaced by "/mnt/hephy", because the scratch disk is wiped periodically.

```
module load anaconda3/2019.10
conda env create --file envs/environment-tf16.yml
conda activate tf16
```
