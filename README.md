# Easy HMM
An easy HMM program written with Python, including the full codes of training, prediction and decoding.
Originally written by tostq, this is just a fork which has translated into Python 3 and added some miscellaneous comments.

# Introduction
- Simple algorithms and models to learn HMMs in pure Python
- Including two HMM models: HMM with Gaussian emissions, and HMM with multinomial (discrete) emissions
- Using unnitest to verify our performance with [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/ "hmmlearn") . 
- Three examples: Dice problem, Chinese words segmentation and stock analysis.

# Code list
- `hmm.py`: hmm models file
- `DiscreteHMM_test.py`, `GaussianHMM_test.py`: test files
- `01_Dice.py`, `02_Stock.py`: example files

# 中文說明
這是分叉自 tostq 大大在 Python 上實作的隱性馬可夫模型（Hidden Markov Model），主要將程式碼改為 Python 3，並加入了一些個人的註解。
原作者 tostq 大大的部落格：[http://blog.csdn.net/tostq/article/details/70846702](http://blog.csdn.net/tostq/article/details/70846702 "hmm")
