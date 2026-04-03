import numpy as np
import h5py
import time
from typing import List, Optional, Dict, Any
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count


### record the function time used
def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_wall = time.time()
        start_cpu = time.process_time()
        result = func(*args, **kwargs)
        end_wall = time.time()
        end_cpu = time.process_time()
        print(func.__name__, "CPU  Time:", end_cpu - start_cpu)
        print(func.__name__, "Wall Time:", end_wall - start_wall)
        return result
    return wrapper

def get_available_fields(fname: str, group: str) -> List[str]:
    """
    Get list of available fields in the specified group of the HDF5 file.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 file
    group : str
        "PartType1" for snapshot particle, "Group" for main halos or "Subhalo" for subhalos
    
    Returns
    -------
    fields : List[str]
        List of available fields in the specified group
    """
    target_group = group
    
    with h5py.File(fname, 'r') as ff:
        if target_group in ff:
            return list(ff[target_group].keys())
        else:
            raise ValueError(f"group '{target_group}' does not exist in the file")

def _parse_file_pattern(fname: str) -> tuple:
    """
    Parse filename pattern to extract base name and number part.
    
    Parameters
    ----------
    fname : str
        Filename with pattern
    
    Returns
    -------
    tuple
        (base_name, number_str, suffix, has_number)
    """
    # Find the last number in the filename
    match = re.search(r'(\d+)\.hdf5$', fname)
    if match:
        number_str = match.group(1)
        base_name = fname[:match.start(1)]
        suffix = fname[match.end(1):]
        return base_name, number_str, suffix, True
    return fname, "", "", False

def _generate_file_list(base_file:str, num_files:int):
    """
    Generate file list based on base file and number of files.
    
    Parameters
    ----------
    base_file : str
        Base filename (e.g., 'halo.0.hdf5')
    num_files : int
        Total number of files
    
    Returns
    -------
    List[str]
        List of generated filenames
    """
    if num_files <= 1:
        return [base_file]
    # Parse pattern
    base_name, number_str, suffix, has_number = _parse_file_pattern(base_file)
    
    return [f"{base_name}{i:0{len(number_str)}d}{suffix}" for i in range(num_files)]
    
    

########################################################################################
############################ Read particle data from HDF5 files ########################
########################################################################################
# @calculate_time
def _get_snapshot_parttype_keys(ff: h5py.File, part_type: int = 1) -> Dict[str, bool]:
    """
    Check which keys are available for a given particle type.
    
    Parameters
    ----------
    ff : h5py.File
        Open HDF5 file
    part_type : int
        Particle type (0=gas, 1=DM, 4=stars, etc.)
    
    Returns
    -------
    Dict[str, bool]
        Dictionary indicating availability of each key
    """
    part_type_str = f"PartType{part_type}"
    available = {}
    
    if part_type_str in ff:
        part_group = ff[part_type_str]
        available['Coordinates'] = 'Coordinates' in part_group
        available['Velocities'] = 'Velocities' in part_group
        available['ParticleIDs'] = 'ParticleIDs' in part_group
    else:
        available['Coordinates'] = False
        available['Velocities'] = False
        available['ParticleIDs'] = False
    
    return available

def IO_Snap_single(fname: str, 
                   return_pos: bool = True, 
                   return_vel: bool = False, 
                   return_id: bool = False,
                   part_type: int = 1,
                   dtype: type = np.float32,
                   id_dtype: type = np.int64) -> Optional[np.ndarray]:
    """
    Read particle data from a single snapshot HDF5 file.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 snapshot file
    return_pos : bool
        Whether to return particle positions
    return_vel : bool
        Whether to return particle velocities
    return_id : bool
        Whether to return particle IDs
    part_type : int
        Particle type (0=gas, 1=DM, 4=stars, etc.)
    dtype : type
        Data type for position and velocity arrays
    id_dtype : type
        Data type for particle IDs
    
    Returns
    -------
    np.ndarray or None
        Structured numpy array containing requested particle data, or None if no data requested
    
    Example
    -------
    data = IO_Snap_single('snapshot_001.hdf5', 
                          return_pos=True, 
                          return_vel=False, 
                          return_id=True,
                          part_type=1,
                          dtype=np.float32,
                          id_dtype=np.int64)
    """
    if not (return_pos or return_vel or return_id):
        print("Warning: No data requested. Set at least one of return_pos, return_vel, or return_id to True.")
        return None
    
    with h5py.File(fname, 'r') as ff:
        part_type_str = f"PartType{part_type}"
        
        if part_type_str not in ff:
            raise ValueError(f"Particle type {part_type} not found in file. "
                           f"Available particle types: {[k for k in ff.keys() if k.startswith('PartType')]}")
        
        # Get particle count for this type
        if 'Header' in ff:
            npart_this_file = ff['Header'].attrs.get('NumPart_ThisFile', np.zeros(6, dtype=np.int64))[part_type]
            npart_total = ff['Header'].attrs.get('NumPart_Total', np.zeros(6, dtype=np.int64))[part_type]
        else:
            # If no header, try to get count from data
            npart_this_file = 0
            for key in ['Coordinates', 'Velocities', 'ParticleIDs']:
                if key in ff[part_type_str]:
                    npart_this_file = ff[part_type_str][key].shape[0]
                    break
        
        if npart_this_file == 0:
            print(f"Warning: No particles of type {part_type} in file {fname}")
            return np.array([], dtype=[])
        
        # Build dtype for structured array
        dtype_list = []
        if return_pos:
            dtype_list.append(("Pos", dtype, (3,)))
        if return_vel:
            dtype_list.append(("Vel", dtype, (3,)))
        if return_id:
            dtype_list.append(("ID", id_dtype))
        
        # Create structured array
        data = np.zeros(npart_this_file, dtype=dtype_list)
        
        # Check which keys are available
        available_keys = _get_snapshot_parttype_keys(ff, part_type)
        
        # Read data
        if return_pos:
            if available_keys['Coordinates']:
                data['Pos'] = ff[f"{part_type_str}/Coordinates"][...].astype(dtype)
            else:
                print(f"Warning: Coordinates not found for PartType{part_type} in {fname}")
                data['Pos'] = np.zeros((npart_this_file, 3), dtype=dtype)
        
        if return_vel:
            if available_keys['Velocities']:
                data['Vel'] = ff[f"{part_type_str}/Velocities"][...].astype(dtype)
            else:
                print(f"Warning: Velocities not found for PartType{part_type} in {fname}")
                data['Vel'] = np.zeros((npart_this_file, 3), dtype=dtype)
        
        if return_id:
            if available_keys['ParticleIDs']:
                data['ID'] = ff[f"{part_type_str}/ParticleIDs"][...].astype(id_dtype)
            else:
                print(f"Warning: ParticleIDs not found for PartType{part_type} in {fname}")
                data['ID'] = np.zeros(npart_this_file, dtype=id_dtype)
        
        return data

def _read_single_snapshot(args: tuple) -> Optional[np.ndarray]:
    """
    Helper function to read a single snapshot file (for multiprocessing).
    
    Parameters
    ----------
    args : tuple
        (filename, return_pos, return_vel, return_id, part_type, dtype, id_dtype)
    
    Returns
    -------
    np.ndarray or None
        Structured array from the file or None if error
    """
    fname, return_pos, return_vel, return_id, part_type, dtype, id_dtype = args
    try:
        return IO_Snap_single(fname, return_pos, return_vel, return_id, part_type, dtype, id_dtype)
    except Exception as e:
        print(f"Error reading snapshot {fname}: {e}")
        return None

@calculate_time
def IO_Snap_multifile(fname: str,
                      return_pos: bool = True,
                      return_vel: bool = False,
                      return_id: bool = False,
                      part_type: int = 1,
                      dtype: type = np.float32,
                      id_dtype: type = np.int64,
                      n_processes: int = 1) -> Optional[np.ndarray]:
    """
    Read particle data from multiple snapshot HDF5 files with multiprocessing.
    
    Parameters
    ----------
    fname : str
        Base snapshot filename (e.g., 'snapshot_0.hdf5')
    return_pos : bool
        Whether to return particle positions
    return_vel : bool
        Whether to return particle velocities
    return_id : bool
        Whether to return particle IDs
    part_type : int
        Particle type (0=gas, 1=DM, 4=stars, etc.)
    dtype : type
        Data type for position and velocity arrays
    id_dtype : type
        Data type for particle IDs
    n_processes : int
        Number of processes for parallel reading.
        If <= 0, use all available CPU cores.
        If = 1, use single process.
    
    Returns
    -------
    np.ndarray or None
        Structured numpy array containing all requested particle data from all files
    
    Example
    -------
    data = IO_Snap_multifile('snapshot_0.hdf5',
                             return_pos=True,
                             return_vel=True,
                             return_id=True,
                             part_type=1,
                             n_processes=4)
    """
    # First, read the first file to get number of files
    try:
        with h5py.File(fname, 'r') as ff:
            if 'Header' in ff and 'NumFilesPerSnapshot' in ff['Header'].attrs:
                num_files = ff['Header'].attrs['NumFilesPerSnapshot']
            elif 'Header' in ff and 'NumFiles' in ff['Header'].attrs:
                num_files = ff['Header'].attrs['NumFiles']
            else:
                num_files = 1
    except Exception as e:
        print(f"Warning: Cannot read header from {fname}, assuming single file: {e}")
        num_files = 1
    
    if num_files == 1:
        # Single file, no multiprocessing needed
        print(f"Reading single snapshot file: {fname}")
        return IO_Snap_single(fname, return_pos, return_vel, return_id, part_type, dtype, id_dtype)
    
    # Generate file list
    file_list = _generate_file_list(fname, num_files)
    
    # Check which files exist
    valid_files = []
    for file_path in file_list:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            print(f"Warning: Snapshot file {file_path} does not exist, skipping")
    
    if not valid_files:
        raise ValueError(f"No valid snapshot files found. Checked {num_files} files based on pattern.")
    
    print(f"Reading {len(valid_files)} snapshot files with {n_processes} processes")
    
    # Determine number of processes
    if n_processes <= 0:
        n_processes = cpu_count()
    elif n_processes > len(valid_files):
        n_processes = len(valid_files)
    
    if n_processes == 1 or len(valid_files) == 1:
        # Single process
        # print("Using single process for snapshot reading...")
        results = []
        for i, file_path in enumerate(valid_files):
            # print(f"Reading snapshot file {i+1}/{len(valid_files)}: {file_path}")
            data = IO_Snap_single(file_path, return_pos, return_vel, return_id, part_type, dtype, id_dtype)
            if data is not None and len(data) > 0:
                results.append(data)
    else:
        # Multiprocessing
        # print(f"Using {n_processes} processes for parallel snapshot reading...")
        pool_args = [(fp, return_pos, return_vel, return_id, part_type, dtype, id_dtype) 
                    for fp in valid_files]
        
        with Pool(processes=n_processes) as pool:
            # Use imap to show progress
            results = []
            for i, result in enumerate(pool.imap(_read_single_snapshot, pool_args), 1):
                if result is not None and len(result) > 0:
                    results.append(result)
                # print(f"Progress: {i}/{len(pool_args)} snapshot files completed")
    
    if not results:
        print("Warning: No data read from any snapshot file")
        return np.array([], dtype=[])
    
    # Concatenate all results
    print(f"Concatenating data from {len(results)} snapshot files...")
    combined_data = np.concatenate(results, axis=0)
    
    # Get total particle count from headers
    total_particles = 0
    for file_path in valid_files:
        with h5py.File(file_path, 'r') as ff:
            if 'Header' in ff:
                npart_total = ff['Header'].attrs.get('NumPart_Total', np.zeros(6, dtype=np.int64))
                total_particles = npart_total[part_type]
                break
    
    print(f"Total {len(combined_data)} particles read (expected {total_particles} from header)")
    
    return combined_data

def get_snapshot_info(fname: str) -> Dict[str, Any]:
    """
    Get information about a snapshot file.
    
    Parameters
    ----------
    fname : str
        Path to the snapshot HDF5 file
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing snapshot information
    """
    with h5py.File(fname, 'r') as ff:
        info = {}
        
        if 'Header' in ff:
            header = ff['Header']
            info['NumFiles'] = header.attrs.get('NumFilesPerSnapshot', header.attrs.get('NumFiles', 1))
            info['NumPart_Total'] = header.attrs.get('NumPart_Total', np.zeros(6, dtype=np.int64))
            info['NumPart_ThisFile'] = header.attrs.get('NumPart_ThisFile', np.zeros(6, dtype=np.int64))
            info['Time'] = header.attrs.get('Time', 0.0)
            info['Redshift'] = header.attrs.get('Redshift', 0.0)
            info['BoxSize'] = header.attrs.get('BoxSize', 0.0)
        
        # List available particle types
        info['AvailablePartTypes'] = [k for k in ff.keys() if k.startswith('PartType')]
        
        # For each particle type, list available datasets
        info['PartTypeDatasets'] = {}
        for part_type in info['AvailablePartTypes']:
            info['PartTypeDatasets'][part_type] = list(ff[part_type].keys())
        
        return info

def get_total_particle_count(fname: str, part_type: int = 1) -> int:
    """
    Get total number of particles of a given type across all snapshot files.
    
    Parameters
    ----------
    fname : str
        Base snapshot filename
    part_type : int
        Particle type
    
    Returns
    -------
    int
        Total number of particles
    """
    # First get number of files
    try:
        with h5py.File(fname, 'r') as ff:
            if 'Header' in ff and 'NumFilesPerSnapshot' in ff['Header'].attrs:
                num_files = ff['Header'].attrs['NumFilesPerSnapshot']
            elif 'Header' in ff and 'NumFiles' in ff['Header'].attrs:
                num_files = ff['Header'].attrs['NumFiles']
            else:
                num_files = 1
    except Exception as e:
        print(f"Warning: Cannot get number of files: {e}")
        num_files = 1
    
    if num_files == 1:
        with h5py.File(fname, 'r') as ff:
            if 'Header' in ff:
                npart_total = ff['Header'].attrs.get('NumPart_Total', np.zeros(6, dtype=np.int64))
                return npart_total[part_type]
            else:
                return 0
    
    # For multiple files, the first file should have the total count
    with h5py.File(fname, 'r') as ff:
        if 'Header' in ff:
            npart_total = ff['Header'].attrs.get('NumPart_Total', np.zeros(6, dtype=np.int64))
            return npart_total[part_type]
        else:
            return 0

########################################################################################
############################ Read halo data from HDF5 files ############################
########################################################################################
# @calculate_time
def IO_Halo_single(fname: str, 
                   readkeys: List[str] = ['GroupPos', 'Group_M_Mean200'], 
                   subhalo: bool = False) -> np.ndarray:
    """
    Read single FOF halo data from HDF5 files and return structured numpy array.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 file
    readkeys : List[str]
        List of attribute names to read.
        For FoF halo: ['GroupPos', 'Group_M_Mean200'] or ['SubhaloPos', 'SubhaloMass']
    subhalo : bool
        Whether to read subhalo data (True for subhalos, False for FoF halos)
    
    Returns
    -------
    halos : np.ndarray
        Structured numpy array containing all requested attributes
    """
    
    with h5py.File(fname, 'r') as ff:
        # Determine target group
        target_group = 'Subhalo' if subhalo else 'Group'
        
        if target_group not in ff:
            raise ValueError(f"Group '{target_group}' not found in file. "
                           f"Available groups: {list(ff.keys())}")
        
        # Get halo count from header
        halo_count = None
        if 'Header' in ff and 'Ngroups_ThisFile' in ff['Header'].attrs:
            if subhalo:
                halo_count = ff['Header'].attrs.get('Nsubhalos_ThisFile', None)
            else:
                halo_count = ff['Header'].attrs.get('Ngroups_ThisFile', None)
        
        # If header count not available, use the first valid attribute's shape
        if halo_count is None:
            for attr in readkeys:
                if attr in ff[target_group]:
                    halo_count = ff[target_group][attr].shape[0]
                    break
        
        if halo_count is None:
            raise ValueError(f"No valid attributes found in group '{target_group}'. "
                           f"Available attributes: {list(ff[target_group].keys())}")
        
        # Build dtype for the structured array
        dtype_list = []
        valid_attrs = []
        
        for attr in readkeys:
            if attr in ff[target_group]:
                data = ff[target_group][attr]
                shape = data.shape
                
                # Get numpy dtype
                if hasattr(data.dtype, 'type'):
                    np_dtype = data.dtype
                else:
                    # Convert HDF5 string type to appropriate numpy type
                    if h5py.check_string_dtype(data.dtype):
                        # For string data, use object dtype
                        max_len = data.dtype.itemsize
                        np_dtype = f'U{max_len}'
                    elif 'float' in str(data.dtype):
                        np_dtype = np.float32
                    elif 'int' in str(data.dtype):
                        # Try to preserve integer type
                        if '64' in str(data.dtype):
                            np_dtype = np.int64
                        else:
                            np_dtype = np.int32
                    else:
                        np_dtype = np.float32
                
                # Determine field shape
                if len(shape) == 1:
                    # 1D data
                    dtype_list.append((attr, np_dtype))
                elif len(shape) == 2:
                    # 2D data, e.g., GroupPos (N,3)
                    dtype_list.append((attr, np_dtype, (shape[1],)))
                elif len(shape) > 2:
                    # 3D or higher dimensional data
                    dtype_list.append((attr, np_dtype, shape[1:]))
                else:
                    # Scalar data
                    dtype_list.append((attr, np_dtype))
                
                valid_attrs.append(attr)
            else:
                print(f"Warning: Attribute '{attr}' not found in group '{target_group}'. Skipping.")
        
        if not dtype_list:
            raise ValueError(f"No valid attributes found. "
                           f"Available in '{target_group}': {list(ff[target_group].keys())}")
        
        # Create and fill structured array
        halos = np.zeros(halo_count, dtype=dtype_list)
        
        for attr in valid_attrs:
            data = ff[target_group][attr]
            
            # Handle different data types
            if h5py.check_string_dtype(data.dtype):
                # String data needs special handling
                str_data = data[()]
                # Convert to unicode strings
                halos[attr] = np.array([s.decode('utf-8') if isinstance(s, bytes) else s 
                                       for s in str_data])
            else:
                # Numerical data
                halos[attr] = data[()]
        
        return halos

def read_both_halo_types(fname: str, 
                         group_attrs: Optional[List[str]] = None,
                         subhalo_attrs: Optional[List[str]] = None) -> tuple:
    """
    Read data from both Group and Subhalo groups.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 file
    group_attrs : List[str], optional
        Attributes to read from Group. If None, read all available attributes.
    subhalo_attrs : List[str], optional
        Attributes to read from Subhalo. If None, read all available attributes.
    
    Returns
    -------
    tuple
        (group_data, subhalo_data) as two structured numpy arrays
    """
    group_data = None
    subhalo_data = None
    
    if group_attrs is None:
        group_attrs = get_available_fields(fname, group='Group')
    
    if subhalo_attrs is None:
        subhalo_attrs = get_available_fields(fname, group='Subhalo')
    
    try:
        group_data = IO_Halo_single(fname, group_attrs, subhalo=False)
    except Exception as e:
        print(f"Warning: Failed to read group data: {e}")
    
    try:
        subhalo_data = IO_Halo_single(fname, subhalo_attrs, subhalo=True)
    except Exception as e:
        print(f"Warning: Failed to read subhalo data: {e}")
    
    return group_data, subhalo_data

def get_halo_count(fname: str, subhalo: bool = False) -> int:
    """
    Get the number of halos/subhalos in the file.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 file
    subhalo : bool
        Whether to get subhalo count (True) or group count (False)
    
    Returns
    -------
    count : int
        Number of halos/subhalos
    """
    with h5py.File(fname, 'r') as ff:
        if 'Header' in ff:
            if subhalo:
                return ff['Header'].attrs.get('Nsubhalos_ThisFile', 0)
            else:
                return ff['Header'].attrs.get('Ngroups_ThisFile', 0)
        else:
            # Try to get count from the first available attribute
            target_group = 'Subhalo' if subhalo else 'Group'
            if target_group in ff and len(ff[target_group].keys()) > 0:
                first_attr = list(ff[target_group].keys())[0]
                return ff[target_group][first_attr].shape[0]
            else:
                return 0

def get_num_files(fname: str) -> int:
    """
    Get number of files from the header.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 file
    
    Returns
    -------
    int
        Number of files
    """
    with h5py.File(fname, 'r') as ff:
        if 'Header' in ff and 'NumFiles' in ff['Header'].attrs:
            return ff['Header'].attrs['NumFiles']
        else:
            return 1

def get_total_halo_count(fname: str, subhalo: bool = False) -> int:
    """
    Get total number of halos/subhalos across all files.
    
    Parameters
    ----------
    fname : str
        Path to the HDF5 file
    subhalo : bool
        Whether to get subhalo count (True) or group count (False)
    
    Returns
    -------
    int
        Total number of halos/subhalos
    """
    num_files = get_num_files(fname)
    
    if num_files == 1:
        with h5py.File(fname, 'r') as ff:
            if 'Header' in ff:
                if subhalo:
                    return ff['Header'].attrs.get('Nsubhalos_Total', 0)
                else:
                    return ff['Header'].attrs.get('Ngroups_Total', 0)
            else:
                return 0
    
    # For multiple files, we need to read all headers
    total_count = 0
    file_list = _generate_file_list(fname, num_files)
    
    for file_path in file_list:
        if Path(file_path).exists():
            with h5py.File(file_path, 'r') as ff:
                if 'Header' in ff:
                    if subhalo:
                        total_count += ff['Header'].attrs.get('Nsubhalos_ThisFile', 0)
                    else:
                        total_count += ff['Header'].attrs.get('Ngroups_ThisFile', 0)
    
    return int(total_count)

def _read_single_file(args: tuple) -> np.ndarray:
    """
    Helper function to read a single file (for multiprocessing).
    
    Parameters
    ----------
    args : tuple
        (filename, readkeys, subhalo)
    
    Returns
    -------
    np.ndarray
        Structured array from the file
    """
    fname, readkeys, subhalo = args
    try:
        return IO_Halo_single(fname, readkeys, subhalo)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return None

@calculate_time
def IO_Halo_multifile(fname: str, 
                      readkeys: List[str] = ['GroupPos', 'Group_M_Mean200'], 
                      subhalo: bool = False,
                      n_processes: int = 1) -> np.ndarray:
    """
    Read multiple FOF halo HDF5 files with multiprocessing support.
    
    Automatically reads NumFiles from Header, then reads all files.
    
    Parameters
    ----------
    fname : str
        Base HDF5 file path (e.g., 'halo_0.hdf5')
    readkeys : List[str]
        List of attribute names to read
    subhalo : bool
        Whether to read subhalo data (True for subhalos, False for FoF halos)
    n_processes : int
        Number of processes for parallel reading.
        If <= 0, use all available CPU cores.
        If = 1, use single process.
    
    Returns
    -------
    halos : np.ndarray
        Structured numpy array containing all requested attributes from all files
    """
    # First, read the first file to get number of files
    try:
        with h5py.File(fname, 'r') as ff:
            if 'Header' in ff and 'NumFiles' in ff['Header'].attrs:
                num_files = ff['Header'].attrs['NumFiles']
            else:
                num_files = 1
    except Exception as e:
        print(f"Warning: Cannot read header from {fname}, assuming single file: {e}")
        num_files = 1
    
    if num_files == 1:
        # Single file, no multiprocessing needed
        print(f"Reading single file: {fname}")
        return IO_Halo_single(fname, readkeys, subhalo)
    
    # Generate file list
    file_list = _generate_file_list(fname, num_files)
    
    # Check which files exist
    valid_files = []
    for file_path in file_list:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            print(f"Warning: File {file_path} does not exist, skipping")
    
    if not valid_files:
        raise ValueError(f"No valid files found. Checked {num_files} files based on pattern.")
    
    print(f"Reading {len(valid_files)} files with {n_processes} processes")
    
    # Determine number of processes
    if n_processes <= 0:
        n_processes = cpu_count()
    elif n_processes > len(valid_files):
        n_processes = len(valid_files)
    
    if n_processes == 1 or len(valid_files) == 1:
        # Single process
        # print("Using single process reading...")
        results = []
        for i, file_path in enumerate(valid_files):
            print(f"Reading file {i+1}/{len(valid_files)}: {file_path}")
            data = IO_Halo_single(file_path, readkeys, subhalo)
            if data is not None and len(data) > 0:
                results.append(data)
    else:
        # Multiprocessing
        # print(f"Using {n_processes} processes for parallel reading...")
        pool_args = [(file_path, readkeys, subhalo) for file_path in valid_files]
        
        with Pool(processes=n_processes) as pool:
            # Use imap to show progress
            results = []
            for i, result in enumerate(pool.imap(_read_single_file, pool_args), 1):
                if result is not None and len(result) > 0:
                    results.append(result)
                # print(f"Progress: {i}/{len(pool_args)} files completed")
    
    if not results:
        print("Warning: No data read from any file")
        return np.array([], dtype=[])
    
    # Concatenate all results
    print(f"Concatenating data from {len(results)} files...")
    combined_data = np.concatenate(results, axis=0)
    print(f"Total {len(combined_data)} halos read")
    
    return combined_data

@calculate_time
def read_both_halo_types_multifile(fname: str, 
                                   group_attrs: Optional[List[str]] = None,
                                   subhalo_attrs: Optional[List[str]] = None,
                                   n_processes: int = 1) -> tuple:
    """
    Read data from both Group and Subhalo groups from multiple files.
    
    Parameters
    ----------
    fname : str
        Base HDF5 file path
    group_attrs : List[str], optional
        Attributes to read from Group. If None, read all available attributes.
    subhalo_attrs : List[str], optional
        Attributes to read from Subhalo. If None, read all available attributes.
    n_processes : int
        Number of processes for parallel reading.
    
    Returns
    -------
    tuple
        (group_data, subhalo_data) as two structured numpy arrays
    """
    if group_attrs is None:
        group_attrs = get_available_fields(fname, subhalo=False)
    
    if subhalo_attrs is None:
        subhalo_attrs = get_available_fields(fname, subhalo=True)
    
    group_data = None
    subhalo_data = None
    
    try:
        group_data = IO_Halo_multifile(fname, group_attrs, subhalo=False, n_processes=n_processes)
    except Exception as e:
        print(f"Warning: Failed to read group data: {e}")
    
    try:
        subhalo_data = IO_Halo_multifile(fname, subhalo_attrs, subhalo=True, n_processes=n_processes)
    except Exception as e:
        print(f"Warning: Failed to read subhalo data: {e}")
    
    return group_data, subhalo_data

