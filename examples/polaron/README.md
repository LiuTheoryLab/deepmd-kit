# Polaron Model

We still use multi-task learning. So one needs to prepare two database, one is for training energy and force, and another is for training spin.

For `database_spin`, there is an additional file: `set.000/atomic_spin.npy`, the shape: `(nframes, natoms * 2)`.