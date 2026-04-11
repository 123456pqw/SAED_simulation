import multem
import numpy as np
import matplotlib.pyplot as plt
from ase import units
from pathlib import Path
from pymatgen.core import Structure,Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
import time
from copy import deepcopy
from skimage.feature import peak_local_max
import matplotlib.colors as colorsOp
import re
from collections import defaultdict


def cif_to_multem_parameters(cif_path, na=8, nb=8, nc=40, ncu=2, zone_axis=[0, 0, 1], rms3d=0.085):
    structure = Structure.from_file(cif_path)
    params = multem.CrystalParameters()
    params.a = structure.lattice.a
    params.b = structure.lattice.b
    params.c = structure.lattice.c
    params.na = na
    params.nb = nb
    params.nc = nc

    atoms = []
    for site in structure:
        frac = site.frac_coords
        symbol = site.species_string
        symbol=  symbol.split(":")[0]
        #print(symbol)
        atoms.append((symbol, frac[0], frac[1], frac[2]))

    if zone_axis != [0,0,1]:
        atoms = _transform_zone_axis(atoms, params, zone_axis)

    layers = _create_ncu_layers(atoms, params, ncu, rms3d)
    params.layers = [multem.AtomList(layer) for layer in layers]

    return params


def _transform_zone_axis(atoms, params, zone_axis):
    h, k, l = zone_axis
    a, b, c = params.a, params.b, params.c

    basis = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
    target = np.array([h / a, k / b, l / c])
    target /= np.linalg.norm(target)

    if not np.allclose(target, [0, 0, 1]):
        v1 = np.array([1, 0, 0], dtype=float) if h == 0 else np.array([-k, h, 0], dtype=float)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(target, v1)
        rot_matrix = np.column_stack([v1, v2, target])
    else:
        rot_matrix = np.eye(3)

    transformed = []
    for symbol, x, y, z in atoms:
        pos = np.array([x * a, y * b, z * c])
        new_pos = rot_matrix.T @ pos
        transformed.append((
            symbol,
            new_pos[0] / a,
            new_pos[1] / b,
            new_pos[2] / c
        ))

    new_basis = rot_matrix.T @ basis
    params.a = np.linalg.norm(new_basis[:, 0])
    params.b = np.linalg.norm(new_basis[:, 1])
    params.c = np.linalg.norm(new_basis[:, 2])

    return transformed


def _create_ncu_layers(atoms, params, ncu, rms3d):
    layer_dict = defaultdict(list)
    c = params.c
    layer_thickness = c / ncu

    for symbol, x, y, z in atoms:
        z_abs = z * c
        layer_idx = int(z_abs / layer_thickness)
        x_frac, y_frac, z_frac = x, y, z

        layer_dict[layer_idx].append((
            element_to_z(symbol),
            x_frac, y_frac, z_frac,
            rms3d, 1.0, 0, 0
        ))

    return [layer_dict[k] for k in sorted(layer_dict.keys())]


def element_to_z(symbol):
    periodict_table = {
        'H': 1,   'He': 2,
        'Li': 3,  'Be': 4,  'B': 5,   'C': 6,   'N': 7,   'O': 8,   'F': 9,   'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,  'S': 16,  'Cl': 17, 'Ar': 18,
        'K': 19,  'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,  'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39,  'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53,  'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92,  'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100,'Md': 101,'No': 102,'Lr': 103,'Rf': 104,'Db': 105,'Sg': 106,
        'Bh': 107,'Hs': 108,'Mt': 109,'Ds': 110,'Rg': 111,'Cn': 112,'Nh': 113,'Fl': 114,
        'Mc': 115,'Lv': 116,'Ts': 117,'Og': 118
    }

    
    #print(periodict_table.get(symbol))
    return periodict_table.get(symbol, 0)


def build_crystal_from_cif(cif_path, na=8, nb=8, nc=40, ncu=2, rms3d=0.085, zone_axis=[0, 0, 1]):
    params = cif_to_multem_parameters(cif_path, na, nb, nc, ncu, zone_axis, rms3d)
    atoms = multem.crystal_by_layers(params)  
    a, b, c = params.a, params.b, params.c
    lx = na * a
    print(na*a,nb*b)
    #ly = nb * b
    ly =na * a
    lz = nc * c

    # 确保 lx 和 ly >= 25 Å
    min_length = 25.0
    if lx < min_length:
        na_new = int(np.ceil(min_length / a))
    else:
        na_new = na
    if ly < min_length:
        nb_new = int(np.ceil(min_length / b))
    else:
        nb_new = nb

    # 如果需要调整，重新生成参数和atoms
    if (na_new != na) or (nb_new != nb):
        params = cif_to_multem_parameters(cif_path, na_new, nb_new, nc, ncu, zone_axis, rms3d)
        atoms = multem.crystal_by_layers(params)
        lx = na_new * params.a
        ly = nb_new * params.b
        lz = nc * params.c

    return atoms, lx, ly, lz, params.a, params.b, params.c, params.c / ncu


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode="constant")

def fourier_interpolation(image, output_size):
    fft = np.fft.fft2(image)
    fft = np.fft.fftshift(fft)
    fft = padding(fft, output_size[0], output_size[1])
    fft = np.fft.ifftshift(fft)
    out_image = np.real(np.fft.ifft2(fft))
    return out_image

def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [keV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """
    energy = energy * 1e3
    return (
        units._hplanck
        * units._c
        / np.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def potential_sampling(energy: float, collection_angle: float) -> float:
    """


    Args:
        energy (float): eV
        collection_angle (float): mrad

    Returns:
        pixel size for potential sampling
    """
    return energy2wavelength(energy) / 3 / (collection_angle / 1e3)

def ceil_to_nearest_even_number(num):
    return int(np.ceil(num / 2) * 2)

def potential_pixel(
    energy: float, collection_angle: float, real_space_length: float
) -> float:
    """
    Args:
        energy (float): eV
        collection_angle (float): mrad
        real_space_length (float): Å

    Returns:
        sampling for potential sampling
    """
    return ceil_to_nearest_even_number(
        real_space_length / potential_sampling(energy, collection_angle)
    )


def plot_image(image, pixel_size, space="real", title=None, filename='runsaed.png', tile=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    fig, ax = plt.subplots()
    if tile is not None:
        image = np.kron(np.ones(tile), image)
    #ax.imshow(image)

    ax.imshow(image, cmap="gray")  # 黑白显示
    
    fontprops = fm.FontProperties(size=18)
    if space == "real":
        unit = "1 Å"
    if space == "reciprocal":
        unit = "1/Å"
    scalebar = AnchoredSizeBar(
        ax.transData,
        1 / pixel_size,
        unit,
        "lower left",
        pad=0.1,
        color="white",
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops,
    )
    #ax.add_artist(scalebar)
    ax.set_yticks([])
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)



def run_ed_simulation(cif_path, zone_axis=(0,0,1), nphonon=20, pn_seed=123456,filename='test.png'):
    #print(f"Loading CIF: {cif_path} with zone axis {zone_axis}")
    
    collection_angle = [("angle1", 5), ("angle2", 10), ("angle3", 20), ("angle4", 30)]
    #collection_angle = [("angle1", 5), ("angle2", 10), ("angle3", 20), ("angle4", 30), ("angle3", 40), ("angle4", 50)]
    bwl: bool = True
    output_size: list = None
    convergence_angle: float=0.6
    #convergence_angle: float=0.3
    energy=200
    system_conf = multem.SystemConfiguration()
    system_conf.precision = "float"
    system_conf.cpu_ncores = 12
    system_conf.cpu_nthread = 1
    system_conf.device = "device"
    system_conf.gpu_device = 0

    input_sim = multem.Input()
    #input_sim.simulation_type = "ED"  # 电子衍射
    input_sim.simulation_type = "CBED"  # 电子衍射
    input_sim.interaction_model = "Multislice"
    input_sim.potential_type = "Lobato_0_12"
    # Potential slicing
    input_sim.potential_slicing = "dz_Proj"
    input_sim.spec_dz = 2.0
    # input_sim.potential_slicing = "Planes"

    #input_sim.detector.type = "Circular"
 

    input_sim.pn_model = "Frozen_Phonon"
    input_sim.pn_coh_contrib = 0
    input_sim.pn_single_conf = False
    input_sim.pn_nconf = nphonon
    #input_sim.pn_dim = 110
    input_sim.pn_dim = 20
    input_sim.pn_seed = 300183

    (
        input_sim.spec_atoms,
        input_sim.spec_lx,
        input_sim.spec_ly,
        input_sim.spec_lz,
        a,
        b,
        c,
        input_sim.spec_dz,
    )  = build_crystal_from_cif(
            cif_path=cif_path,
            na=4, nb=4, nc=8, ncu=10, zone_axis=list(zone_axis)
    )


    '''
    na = 8
    nb = 8
    nc = 40
    ncu = 2
    rms3d = 0.085
    (
        input_sim.spec_atoms,
        input_sim.spec_lx,
        input_sim.spec_ly,
        input_sim.spec_lz,
        a,
        b,
        c,
        input_sim.spec_dz,
    ) = Si001_crystal(na, nb, nc, ncu, rms3d)
    '''

    # Specimen thickness
    input_sim.thick_type = "Through_Thick"
    #input_sim.thick = [x * input_sim.spec_dz for x in range(0, 100)]
    input_sim.thick = [x * input_sim.spec_dz for x in range(0, 10)]

    # Microscope parameters
    input_sim.E_0 = energy
    input_sim.theta = 0.0
    input_sim.phi = 0.0

    # Illumination model
    input_sim.illumination_model = "Coherent"
    input_sim.temporal_spatial_incoh = "Temporal_Spatial"

    # Set the incident wave
    input_sim.iw_type = "Auto"
    # input_sim.iw_psi = read_psi_0_multem(input_sim.nx, input_sim.ny)
    input_sim.iw_x = [input_sim.spec_lx / 2]  # input_sim.spec_lx/2
    input_sim.iw_y = [input_sim.spec_ly / 2]  # input_sim.spec_ly/2

    input_sim.cond_lens_m = 0
    input_sim.cond_lens_c_10 = 0
    input_sim.cond_lens_c_30 = 0
    input_sim.cond_lens_c_50 = 0.00
    input_sim.cond_lens_c_12 = 0.0
    input_sim.cond_lens_phi_12 = 0.0
    input_sim.cond_lens_c_23 = 0.0
    input_sim.cond_lens_phi_23 = 0.0
    input_sim.cond_lens_inner_aper_ang = 0.0
    input_sim.cond_lens_outer_aper_ang = convergence_angle

    # defocus spread function
    ti_sigma = multem.iehwgd_to_sigma(32)
    input_sim.cond_lens_ti_a = 1.0
    input_sim.cond_lens_ti_sigma = ti_sigma
    input_sim.cond_lens_ti_beta = 0.0
    input_sim.cond_lens_ti_npts = 5

    # Source spread function
    si_sigma = multem.hwhm_to_sigma(0.45)
    input_sim.cond_lens_si_a = 1.0
    input_sim.cond_lens_si_sigma = si_sigma
    input_sim.cond_lens_si_beta = 0.0
    #input_sim.cond_lens_si_rad_npts = 8
    #input_sim.cond_lens_si_azm_npts = 412
    input_sim.cond_lens_si_rad_npts = 4
    input_sim.cond_lens_si_azm_npts = 4
    # Zero defocus reference
    input_sim.cond_lens_zero_defocus_type = "First"
    input_sim.cond_lens_zero_defocus_plane = 0
    
    max_collection_angle = max(collection_angle, key=lambda item: item[1])[1]
    '''
    input_sim.nx = potential_pixel(
        energy=energy,
        collection_angle=max_collection_angle,
        real_space_length=input_sim.spec_lx,
    )
    input_sim.ny = potential_pixel(
        energy=energy,
        collection_angle=max_collection_angle,
        real_space_length=input_sim.spec_ly,
    )
    '''
    input_sim.nx = 1024
    input_sim.ny = 1024
    #input_sim.nx = 640
    #input_sim.ny = 640
    input_sim.bwl = bwl
    print(1)
    output_multislice = multem.simulate(system_conf, input_sim)
    print(2)
    data = []
    for i in range(len(output_multislice.data)):
            m2psi_tot = output_multislice.data[i].m2psi_tot
            if output_size is not None:
                m2psi_tot = fourier_interpolation(m2psi_tot, output_size)
            data.append(np.array(m2psi_tot))
    max_collection_angle_in_A = (
            max_collection_angle / energy2wavelength(energy) / 1e3 #
    )
    pixel_size = max(
            max_collection_angle_in_A * 2 / input_sim.nx,
            max_collection_angle_in_A * 2 / input_sim.ny,
    )
    pixel_size = max(
            max_collection_angle_in_A * 1 / input_sim.nx,
            max_collection_angle_in_A * 1 / input_sim.ny,
    )
    #print(data)
    cbed_avg = np.abs(sum(data))/len(data)
    #epsilon = 1e-8
    epsilon = 1e-8
    safe_image = np.log(cbed_avg + epsilon)  # 避免log(0)
    '''
    # 选择只绘制中间较小区域，裁剪掉边缘，防止边缘伪影影响
    h, w = safe_image.shape
    crop = 512
    safe_image_crop = safe_image[(h//2 - crop//2):(h//2 + crop//2), (w//2 - crop//2):(w//2 + crop//2)]
    '''
    plot_image(safe_image, pixel_size=pixel_size, space="reciprocal",filename=filename)

if __name__ == "__main__":
    # 这里替换成你的cif文件路径和带轴方向
    cif_file = ""
    zone = (2, 1, 1)
    run_ed_simulation(cif_file, zone_axis=zone, nphonon=20)
