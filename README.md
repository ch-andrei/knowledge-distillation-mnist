# Reproducing Knowledge Distillation on MNIST

See [Reproducing_Knowledge_Distillation_on_MNIST.pdf](Reproducing_Knowledge_Distillation_on_MNIST.pdf) for more details.

trainer.py provides all the training logic, networks, etc.
plotter.py makes all the plots.

To reproduce the results:
1. run "train.sh" to do all the experiments (this step takes 1-2 hours)
2. run "train_800_20_kd.sh" to do the initial simple experiment
3. run "train_800_20_kd_no3s.sh" to do the initial simple experiment with 3s removed from the distilled set
4. run "python plotter.py" to plot all the results and save figures
