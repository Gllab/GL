# MF-MindSpore

This repository is an official MindSpore implementation of our papers:

1.  "A Differential Evolution-Enhanced Position-Transitional Approach to Latent Factor Analysis". (*IEEE Transactions on Emerging Topics in Computational Intelligence, 2022*). [[download](https://ieeexplore.ieee.org/abstract/document/9839514)]
2. "Constraint-Induced Symmetric Nonnegative Matrix Factorization for  Accurate Community Detection". (*Information Fusion, 2023*). [[download](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001300)]

3. "Symmetry and nonnegativity-constrained matrix factorization for community detection". (*IEEE/CAA Journal of Automatica Sinica, 2022*). [[download](https://ieeexplore.ieee.org/abstract/document/9865020)]


## Prerequisites:

1. MindSpore 2.1.1
2. numpy
3. sklearn


## Dataset

All datasets used in Paper 1 are follows.

|  Datasets  | $|U|$  |  $|I|$  | |Lambda| | Data Density |
| :--------: | :----: | :-----: | :----------: | :----------: |
|   ML10M    | 12,547 | 35,250  |      4       |    1.31%     |
| ExtEpinion | 11,023 | 280,755 |      13      |    0.015%    |
|  Flixter   | 7,181  | 253,820 |      30      |    0.11%     |
|   Douban   | 11,751 | 270,667 |      5       |    0.22%     |

All datasets used in Paper 2 are follows.

|  Datasets   | Nodes  |  Edges  | Communities |       Description       |
| :---------: | :----: | :-----: | :---------: | :---------------------: |
|    DBLP     | 12,547 | 35,250  |      4      |   DBLP collaboration    |
| Friendster  | 11,023 | 280,755 |     13      |    Friendster online    |
| LiveJournal | 7,181  | 253,820 |     30      |   LiveJournal online    |
|    Orkut    | 11,751 | 270,667 |      5      |      Orkut online       |
|   Youtube   | 11,144 | 36,186  |     40      |     Youtube online      |
|  Polblogs   | 1,490  | 16,718  |      2      | Blogs about US politics |

All datasets used in Paper 3 are follows.

|  Datasets   | Nodes  |  Edges  | Communities |    Description     |
| :---------: | :----: | :-----: | :---------: | :----------------: |
|   Youtube   | 11,144 | 36,186  |     40      |   Youtube online   |
| Friendster  | 11,023 | 280,755 |     13      | Friendster online  |
| LiveJournal | 7,181  | 253,820 |     30      | LiveJournal online |
|    Orkut    | 11,751 | 270,667 |      5      |    Orkut online    |
|   Amazon    |  5304  |  16701  |     85      |   Amazon product   |
|    DBLP     | 12,547 | 35,250  |      4      | DBLP collaboration |

## Results

### "A Differential Evolution-Enhanced Position-Transitional Approach to Latent Factor Analysis"



### "Constraint-Induced Symmetric Nonnegative Matrix Factorization for  Accurate Community Detection"

![image-20231009204754031](C:\Users\hu\AppData\Roaming\Typora\typora-user-images\image-20231009204754031.png)

### "Symmetry and nonnegativity-constrained matrix factorization for community detection"

![ea118b6f1edee11a02624ac2bbe3d31](D:\Tencent\WeChat\Documents\WeChat Files\huqicongya\FileStorage\Temp\ea118b6f1edee11a02624ac2bbe3d31.png)



## Citation

If you find our papers useful in your research, please consider citing:

```
@article{chen2022differential,
  title={A differential evolution-enhanced position-transitional approach to latent factor analysis},
  author={Chen, Jia and Wang, Renfang and Wu, Di and Luo, Xin},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  volume={7},
  number={2},
  pages={389--401},
  year={2022},
  publisher={IEEE}
}

@article{liu2023constraint,
  title={Constraint-Induced Symmetric Nonnegative Matrix Factorization for Accurate Community Detection},
  author={Liu, Zhigang and Luo, Xin and Wang, Zidong and Liu, Xiaohui},
  journal={Information Fusion},
  volume={89},
  pages={588--602},
  year={2023},
  publisher={Elsevier}
}

@article{liu2022symmetry,
  title={Symmetry and nonnegativity-constrained matrix factorization for community detection},
  author={Liu, Zhigang and Yuan, Guangxiao and Luo, Xin},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={9},
  number={9},
  pages={1691--1693},
  year={2022},
  publisher={IEEE}
}
```

## 
