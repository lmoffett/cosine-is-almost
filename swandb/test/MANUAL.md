# Running Manual Tests

## Sweeps
- Test training using method launcher: `swandb sweep launch test/mock_method_sweep.yaml`
- Test training using the file launcher: `swandb sweep launch test/mock_command_sweep.yaml`
- Test training with gpu limits: `swandb sweep launch test/mock_time_limit_sweep.yaml`
- Test on slurm: `swandb sweep launch test/mock_method_sweep.yaml --runner-config=test/mock_slurm.yaml`

To debug the training sample directly, run `WANDB_OFFLINE=true /usr/bin/env python test/mock_train.py --baseline=300 --param1=3 --param2=30`.

Note that `test/test_main.py` tests the method launcher training.
It is equivalent to `python -m swandb sweep train test/mock_train.py::train param1=3 param2=30 baseline=300`.

## Runs

