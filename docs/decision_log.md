# Decision log

Every choice that changes data, settings or results, with the reason and what
to do if it turns out wrong. The Methods chapter is written from this file.
The same entries are mirrored in the project tracker.

Format: **what** was decided, **why**, and the **fallback** if it is wrong.

---

## 2026-11-13

**D01. Following plan v2 (13 Nov).** Label: 80+ overall within 3 versions, age 21 or under.
*Why:* the design is fixed before any data is modelled.
*Fallback:* plan section 9 (threshold 78 or 75, or age 23 and under), to be decided in week 1 before any model result.

**D02. Environment: the existing `tf-gpu-env` virtual environment (Python 3.12.16), not a new conda environment.**
Added lightgbm 4.7.0, optuna 5.0.0, shap 0.54.1, seaborn 0.13.2, statsmodels 0.15.0 and pyarrow 26.0.1; nothing already installed was changed.
*Why:* the environment already has a working TensorFlow-GPU stack for the RTX 3050, and a second CUDA stack would risk breaking it.
*Fallback:* `requirements.txt` pins the versions, so a clean environment can be built from it.

**D03. The MLP is written in TensorFlow/Keras and trained on the GPU; LightGBM, the random forest and logistic regression run on the CPU.**
*Why:* the plan allows "PyTorch or scikit-learn" for the MLP; Keras is the GPU framework this environment is built around, and it supports the same weighted loss and early stopping. LightGBM's GPU learners are not usable here: the pip wheel has no CUDA tree learner, WSL2 exposes no OpenCL device, and there is no compiler to rebuild it. With about 12,000 training rows, CPU histogram training takes well under a second anyway.
*Fallback:* the MLP class has a scikit-learn style interface, so a PyTorch version could replace it without touching the notebooks.

**D04. LightGBM needs the OpenMP runtime `libgomp.so.1`, which is not installed system-wide.**
The Ubuntu package `libgomp1` was unpacked into `tf-gpu-env/lib/openmp` and that folder added to `LD_LIBRARY_PATH` in the environment's `activate` script.
*Why:* no sudo access on this machine.
*Fallback:* `sudo apt install libgomp1` makes the workaround unnecessary.

**D05. Data source: Kaggle "EA Sports FC 24 Complete Player Dataset", `male_players.csv`, downloaded with `scripts/download_data.py`.**
The file has 180,021 rows and 109 columns covering FIFA 15 to FC 24, as the plan assumed.
*Why:* public dataset, no login needed, the download is scripted so it can be repeated.
*Fallback:* none needed.
