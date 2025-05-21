from deepmd.infer import DeepPot
from deepmd.infer.deep_polaron import DeepPolaron
import dpdata
import numpy as np
import os

# List of folders to process

folders = ['database_population_train']

# 2. Load the model

model = 'model.ckpt.pt'
model_ener = DeepPot(model, head='ener_force')
model_spin = DeepPolaron(model, head='spin')

# 3. Test the model on each folder

for idx, folder in enumerate(folders):
    print(f"Processing folder: {folder}")

    # 1. Load the database

    system = dpdata.LabeledSystem(folder, fmt='deepmd/npy')
    coords = system['coords']
    cells = system['cells']
    atomic_types = system['atom_types']

    nframes = coords.shape[0]
    natoms = coords.shape[1]

    # 3.1 aparam: all zeros

    aparam_zeros = np.zeros((nframes, natoms, 1))

    ener, force, _, _, _, = model_ener.eval(
        coords=coords,
        cells=cells,
        atom_types=atomic_types,
        atomic=True,
        aparam=aparam_zeros

    )

    spin = model_spin.eval(
        coords = coords,
        cells = cells,
        atom_types=atomic_types,
        aparam = aparam_zeros
    )[0] # spin is the first element of the tuple

    np.save(f'0_{idx}_ener.npy', ener)
    np.save(f'0_{idx}_force.npy', force)
    np.save(f'0_{idx}_spin.npy', spin)

    print(f"Energy evaluation with all zeros aparam completed for {folder}")

    # 3.2 aparam: read from file

    aparam_file_path = os.path.join(folder, 'set.000/aparam.npy')
    if os.path.exists(aparam_file_path):
        aparam = np.load(aparam_file_path)

        ener, force, _, _, _, = model_ener.eval(
            coords=coords,
            cells=cells,
            atom_types=atomic_types,
            atomic=True,
            aparam=aparam
        )

        spin = model_spin.eval(
        coords = coords,
        cells = cells,
        atom_types=atomic_types,
        aparam = aparam_zeros
        )[0]

        np.save(f'1_{idx}_ener.npy', ener)
        np.save(f'1_{idx}_force.npy', force)
        np.save(f'1_{idx}_spin.npy', spin)

        print(f"Energy evaluation with aparam from file completed for {folder}")
    else:
        print(f"aparam file not found at {aparam_file_path}")

