# PrivacyFlow

## Requirements
* Python 3.10.4
* (Optional) Cuda 11.8
* pip or conda (recommended: pip)

To install the required Python packages run the following code in the terminal.
```
pip install -r requirements.txt
```

## Code Structure

The module `privacyflow` contains several python classes, which are used inside the notebooks.
The main components are configs, datasets, models and preprocessing functionality.
Some empty folders (containing only a .gitkeep) are also located inside this module, which are used for saving data.

There are 4 Jupyter notebooks, which contain executable code, where different models are trained and evaluated.
The `Differential_Privacy_*.ipynb` notebooks contain code for training models with and without DPSGD. 
These models are used inside the `Membership_Inference_Attack.ipynb` notebook and the `Model_Inversion_Notebook.ipynb` to evaluate attacks against these models.
```
├── privacyflow
│   ├── configs
│   ├── datasets
│   ├── models
│   ├── preprocessing
├── Differential_Privacy_Cifar10.ipynb
├── Differential_Privacy_Face_Models.ipynb
├── Membership_Inference_Attack.ipynb
├── Model_Inversion_Attack.ipynb
```