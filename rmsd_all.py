import subprocess
from ase.io import read, write
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import glob

input_extxyz = 'all_minima_no_duplicates.extxyz'
reference_file = 'min000000.extxyz'
output_rmsd = 'rmsd.txt'
stderr_log = 'stderr.log'
max_workers=20

# Step 1: Read and write individual XYZ files (sequential)
structures = read(input_extxyz, index=':')

xyz_files = []
for i, atoms in enumerate(structures, start=1):
    xyz_filename = f's{i}.xyz'
    write(xyz_filename, atoms)
    xyz_files.append(xyz_filename)
    print(f'{xyz_filename} written')

# Step 2: Define a function for parallel RMSD calculation
def compute_rmsd(xyz_file):
    result = subprocess.run(
        ['rmsdFinder', '-A', xyz_file, '-B', reference_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return (xyz_file, result.stdout, result.stderr)

# Step 3: Run RMSD calculations in parallel
with open(output_rmsd, 'w') as fout, open(stderr_log, 'w') as ferr:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_rmsd, xyz): xyz for xyz in xyz_files}
        
        for future in as_completed(futures):
            xyz_file, stdout, stderr = future.result()
            fout.write(stdout)
            if stderr:
                ferr.write(f'Error in {xyz_file}:\n{stderr}\n')
            print(f'{xyz_file} Finished')

# Step 4: Cleanup generated XYZ files
for file in glob.glob('s*.xyz'):
    os.remove(file)
    print(f'{file} deleted')

print('All tasks completed and temporary files removed.')
