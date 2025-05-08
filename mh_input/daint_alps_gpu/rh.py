import os
import torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
#local_rank = int(os.environ.get("SLURM_LOCALID", 0))
#torch.cuda.set_device(local_rank % torch.cuda.device_count())
import numpy as np
from ase.constraints import FixAtoms
import sys
from ase.io import read, write
from ase import build
from minimahopping.minhop import Minimahopping
from ase.cluster.wulff import wulff_construction
import logging
# from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
from mace.calculators import mace_mp
from ase import build

from mpi4py import MPI
        
def run_minima_hopping(**kwargs):
    # Read the input structure
    initial_configuration = read("input.cif")
    # Fix atoms
    cell = initial_configuration.get_cell()
    inv_cell = np.linalg.inv(cell.T)
    frac_coords = initial_configuration.get_positions() @ inv_cell
    indices_to_fix = [i for i, pos in enumerate(frac_coords) if pos[2] < 0.1919191919191919191919]
    cc = [FixAtoms(indices=indices_to_fix)]    
    # Set up the calculator
    
    try:
        mace_model = '/users/zxingfan/bin/mace_up/mace-omat-r2scan.model'
        calculator = mace_mp(model=mace_model, dispersion=False, default_dtype="float64", device='cuda') #, enable_cueq=True)
    except Exception as e:
        print(f"Failed to initialize calculator for the model {mace_model}: {e}")
        exit()

    initial_configuration.calc = calculator

    # Default parameters
    default_params = {
        'verbose_output': True,
        'constraints': cc,
        'fingerprint_threshold': 1e-3,
        'use_MPI': True,
        'enhanced_feedback': False,
        'mdmin': 2,
        'n_soft': 100,
        'soften_positions': 3e-2,
        'fmax': 1e-3,
        'n_S_orbitals': 1,
        'n_P_orbitals': 0,
        'width_cutoff': 4.0,
        'collect_md_data': False,
        'write_graph_output': True,
        'energy_threshold': 1e-4,
        'T0': 100,
        'Ediff0': 0.1,
        'beta_increase': 1.1,
        'beta_decrease': 0.91,
        'alpha_reject': 1.05,
        'alpha_accept': 0.95,
        'run_time': '0-22:00:00',
        'T0': 100,
        'logLevel' : logging.DEBUG,
        'fixed_cell_simulation' : True
    }

    # Update defaults with provided kwargs
    params = {**default_params, **kwargs}

    with Minimahopping(initial_configuration, **params, ) as mh:
        mh(totalsteps=1000000)

def main():  
    run_minima_hopping()   


if __name__ == '__main__':
    main()
    quit()

