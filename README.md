Basic version of Conv, BN and ReLU layer in `basic` folder, just implement the forward method. \
And using Fuse Conv+BN method and try to imporve in `imporve` folder.

## Basic settings

#### input tensor

N, C, H, W

#### NN structure for testing

> input tensor: \
N = 20
C = 3
H * W = 128 * 128

> network structure\
conv(16, kernel size: 3*3) + bn + relu + \
conv(32, kernel size: 3*3) + bn + relu

> compare with the result from pytorch API

## Imporved method

Fused batch normalization: \
* To calculate var, use the equation `var = E(x^2) - E(x)^2`
* Combine the conv and batch layer together.



## Printout result

```
customized NN running time:  16.252690315246582
conv diff:  tensor(9.5026e-10, dtype=torch.float64)
bn diff:  tensor(0.0418, dtype=torch.float64)
relu diff:  tensor(0.0220, dtype=torch.float64)
conv2 diff:  tensor(0.2359, dtype=torch.float64)
bn2 diff:  tensor(0.0540, dtype=torch.float64)
relu2 diff:  tensor(0.0276, dtype=torch.float64)
pytorch NN running time:  0.06016802787780762
```

```
customized NN (improved) running time:  15.82162594795227
conv bn diff:  tensor(0.0418, dtype=torch.float64)
relu diff:  tensor(0.0220, dtype=torch.float64)
convbn2 diff:  tensor(0.0540, dtype=torch.float64)
relu2 diff:  tensor(0.0276, dtype=torch.float64)
pytorch NN running time:  0.04849410057067871
```

## Conclusion

Seems not much improvement, and sometimes even worse.
The most time consuming part is Conv Layer acutally, thus combine Conv and BN seems not help much.

## Reference
https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post155s2-file3.pdf