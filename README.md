# FYS-STK3155 Project 3
By Severin Schirmer, Tobias Laundal and Didrik Sten Ingebrigtsen

This project aims to solve the regression problem of estimating the mass of a particle based on how it behaves in a gravity field. The data is synthetic, and we use several neural network architectures as models. In the end, we achieve R^2 scores of roughly 0.45, demonstrating that the task is possible to solve partially with relatively simple models.

The project report is in the root folder, and the python scripts and iPython notebooks can be found in the `/src` folder. The latex files and plots are in the `/latex` folder. 

## Quick breakdown of the modules in `/src`
- `FFNN_models.py` houses the CNN and DNN models, as well as utility methods for training and testing them.
- `FFNNs.ipynb` is the notebook where the model is trained and tested, and experimented with.
- `Simple_FFNN.ipynb` uses the same models on an easier version of the problem, that we ended up not needing because we found a clever way of embedding our temporal data in a better way. It can still be interesting to look at for validation, and for understanding the limits of these models, but is not a central part of what our project ended up being, and is not covered in the report.
- `crnn.py` and `crnn.ipynb` has the same purpose as `FFNN_models.py` and `FFNNs.ipynb` respectively, but implement RNN and CRNN models.
- `dataset.py` uses the simulation and field generation functions to build a dataset. This is used by the CNN and DNN models, but not the RNN and CRNN models who generate data on demand. The difference is there simply because of a time constraint and different people being responsible for the different parts.
- `demo_simulation.ipynb` makes some plots for demonstrating how the simulation and field generation work.
- `fields.py` handles generation of scalar potential fields, based on perlin noise. It also has functions for finding the force field from its gradient, and downsizing it to prepare it for being used.
- `loss_visualisation.py` plots out the three different loss functions in two dimensions, since our problem is 1D and seeing how the loss is a function of target and prediction is therefore very useful.
- `simulation.py` simulates how a particle of a given mass moves around in a field. It also has a function for encoding the position into an arrray of the same dimensions as the field.
- `test_fields.py` and `test_simulation.py` has some test functions for verifying that `fields.py` and `simulation.py` functions properly.
