# Colab cells (paste one per cell)

```python
# 1. GPU + versions (record for the report; screenshot nvidia-smi)
!nvidia-smi
import torch, sys; print(torch.__version__, torch.version.cuda, sys.version)
```
```python
# 2. Drive + repo
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ['CMPE260_ROOT'] = '/content/drive/MyDrive/cmpe260_project1'
!git clone <YOUR_REPO_URL> /content/cmpe260-project1 && cd /content/cmpe260-project1 && pip install -q -r requirements.txt
```
```python
# 3. Smoke test (~2 min): proves the whole pipeline before burning compute
%cd /content/cmpe260-project1
!python -m pytest -q tests
!python train.py --policy egreedy --replay uniform --seed 42 --run-name smoke --max-frames 20000
```
```python
# 4. Real run (re-run this same cell after a disconnect: --resume picks up latest.pt)
!python train.py --policy egreedy --replay uniform --seed 42 --run-name egreedy_uniform_s42 --resume
```
```python
# 5. Live curves
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/cmpe260_project1/runs
```
