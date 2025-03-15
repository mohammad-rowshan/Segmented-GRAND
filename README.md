# Segmented ORB-GRAND for eBCH code (128,106)
If you find this algorithm useful, please cite the following paper. Thanks.

M. Rowshan and J. Yuan, "Low-Complexity GRAND by Segmentation," GLOBECOM 2023 - 2023 IEEE Global Communications Conference, Kuala Lumpur, Malaysia, 2023, pp. 6145-6151, doi: 10.1109/GLOBECOM54140.2023.10436895.

[https://ieeexplore.ieee.org/abstract/document/9328621](https://ieeexplore.ieee.org/abstract/document/10436895/)

The extended version can be found here: [https://arxiv.org/abs/2305.14892](https://ieeexplore.ieee.org/abstract/document/10879547)

Abstract: The recently introduced maximum-likelihood (ML) decoding scheme called guessing random additive noise decoding (GRAND) has demonstrated a remarkably low time complexity in high signal-to-noise ratio (SNR) regimes. However, the complexity is not as low at low SNR regimes and low code rates. To mitigate this concern, we propose a scheme for a near-ML variant of GRAND called ordered reliability bits GRAND (or ORBGRAND), which divides codewords into segments based on the properties of the underlying code, generates sub-patterns for each segment consistent with the syndrome (thus reducing the number of inconsistent error patterns generated), and combines them in a near-ML order using two-level integer partitions of logistic weight. The numerical evaluation demonstrates that the proposed scheme, called segmented ORBGRAND, reduces the average number of queries (time complexity/latency) to one-third at all SNR regimes. Moreover, the segmented ORBGRAND with abandonment also improves the error correction performance.

Description: 
This is an implementation of the segmented GRAND algorithm with a simulator for eBCH code (128,106) in Python. The two-level integer partitioning functions can be used for any code. 
The simulator provides additional statistics such as time per query and decoding, average number of queries per decoding, etc.

The algorithm might be confusing. Please read the paper carefully, and if you still do not understand it, feel free to contact me via email address mrowshan at ieee dot org

I will also share the scripts for Segmented ORB-GRAND for PAC codes here soon.

You can find the script for conventional ORBGRAND and the constrained ORBGRAND (all in one file) at https://github.com/mohammad-rowshan/Constrained-Error-Pattern-Generation-for-GRAND which is based on the following paper:

M. Rowshan and J. Yuan, "Constrained Error Pattern Generation for GRAND," 2022 IEEE International Symposium on Information Theory (ISIT), Espoo, Finland, 2022, pp. 1767-1772, doi: 10.1109/ISIT50566.2022.9834343.

[https://ieeexplore.ieee.org/document/9354542](https://ieeexplore.ieee.org/abstract/document/9834343)

Please report any bugs to mrowshan at ieee dot org
