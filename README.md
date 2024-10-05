# angleTL: Angle-based Transfer Learning in High Dimensions

**angleTL** is an R package designed for flexible and robust transfer learning in high-dimensional settings. Transfer learning improves the performance of a target model by incorporating information from related source populations. In many practical scenarios, only parameter estimates from pre-trained source models are available. This package implements a novel angle-based transfer learning approach that leverages the concordance between source and target model parameters, as introduced in:

> **Gu T, Han Y, Duan R.** “Robust angle-based transfer learning in high dimensions” [ArXiv](https://arxiv.org/abs/2210.12759)


## Installation

To install the development version of **angleTL** from GitHub, use the following command in R:

```r
# Install the devtools package if you haven't already
# install.packages("devtools")

devtools::install_github('biostat-duan-lab/angleTL')
```

## Usage Example

Once installed, you can start using **angleTL** with:

```r
library(angleTL)
out = angleTL(X, y, w.src)
```

For detailed documentation, visit the [package documentation](https://github.com/biostat-duan-lab/angleTL).

## Citation

If you use **angleTL** in your research, please cite the following paper:

**Gu T, Han Y, Duan R.** “Robust angle-based transfer learning in high dimensions” [ArXiv](https://arxiv.org/abs/2210.12759)
