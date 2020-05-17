# LiNGAM-GC

Python code of causal discovery algorithm proposed in

[Causality in linear nongaussian acyclic models in the presence of latent gaussian confounders](https://www.mitpressjournals.org/doi/abs/10.1162/NECO_a_00444?casa_token=CeCMzGI9fKsAAAAA:2Wz80tIpYEDxIw-1Rpo5M4KHqkysdLpfE6hd7mpIz11ERrfNgSAxamHNfc-247OudM4u4dSISj1B)  
Chen, Zhitang, and Laiwan Chan  
*Neural Computation* 25.6 (2013): 1605-1641.

## Prerequisites

- numpy
- scipy
- sklearn
- copy

We test the code using python 3.6.8 on Windows 10. Any later version should still work perfectly.

## Running the test

After installing all required packages, you can run *demo.py* to see whether **LiNGAM-GC** could work normally.

The test code does the following:

1. it generates 10,000 observations (a (10,000, 4) *numpy array*) from a causal model with 4 variables;
2. it applies LiNGAM-GC to the generated data to infer the true causal graph.

## Apply **LiNGAM-GC** on your data

### Usage

```python
mdl = LiNGAM_GC()
mdl.fit(X)
```

Detailed instructions on the usage is given in *lingamgc.py*

## Author

- **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/LiNGAM-GC/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
