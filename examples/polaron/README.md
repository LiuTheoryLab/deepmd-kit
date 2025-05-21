# Polaron Model

We still use multi-task learning. So one needs to prepare two database, one is for training energy and force, and another is for training spin.

For `database_spin`, there is an additional file: `set.000/atomic_spin.npy`, the shape: `(nframes, natoms * 2)`.

## How to use Python interface

```python
# Load the model
from deepmd.infer.deep_polaron import DeepPolaron
model = 'model.ckpt.pt'
model_spin = DeepPolaron(model, head='spin')

# Load the database
system = dpdata.LabeledSystem(folder, fmt='deepmd/npy')
coords = system['coords']
cells = system['cells']
atomic_types = system['atom_types']

# Load the aparam
aparam_file_path = os.path.join(folder, 'set.000/aparam.npy')
aparam = np.load(aparam_file_path)

# eval 
spin = model_spin.eval(
        coords = coords,
        cells = cells,
        atom_types=atomic_types,
        aparam = aparam_zeros
        )[0] # the first element is the spin
```