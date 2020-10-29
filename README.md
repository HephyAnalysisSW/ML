# Machine learning outside CMSSW

## Conda on hepgpu01.hephy.oeaw.ac.at:
```
source /local/HephyML/setup.sh
conda env create --file environment-tf16.yml
conda activate tf16
```

## Conda on cbe.vbc.ac.at:
Make a ~/.condarc file with:
```
auto_activate_base: false
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
