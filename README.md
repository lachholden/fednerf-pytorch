This repository is presented as a demonstration program for the Federated NeRF paper [arXiv:2305.01163](https://arxiv.org/abs/2305.01163).

It is forked from the [krrish94/nerf-pytorch](https://github.com/krrish94/nerf-pytorch)
repository.

The key contribution is the impelementation of Algorithm 2 from the paper in
`train_nerf_federated.py` The update compression implementation can be found in
`update_compression.py`.

The `lego` dataset is included as a demonstration. You can run the demonstration program
as follows:
```shell
$ pip install -r requirements.txt
$ python train_nerf_federated.py --config demonstration/lego.yml 
```

The trained networks can then be evaluated, generating CSV files of validation metrics
and rendered images, by running
```shell
$ ./demonstration/process_results.sh ./demonstration_results/lego 19999 04999
```
