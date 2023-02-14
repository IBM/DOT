import sys
import json

import fenics as fem

# import mshr
import numpy as np
import math 

import scipy
import scipy.sparse as sp
import scipy.optimize as spo
from scipy.interpolate import RectBivariateSpline
import scipy.interpolate as spi
import scipy.io
import scipy.signal as sps
from scipy.ndimage import gaussian_filter

import cvxpy as cp
import mosek 

import time
from tqdm import trange, tqdm
import logging
import os
import sys
import time

from src.utils import *


results_dir = '.'
if len(sys.argv) > 1:
    results_dir = sys.argv[1]
    
with open(results_dir+'/problem.json', 'r') as fp:
    problem = json.load(fp)
with open(results_dir+'/solver_params.json', 'r') as fp:
    solver_params = json.load(fp)

# logging into terminal
name = solver_params['name']+str(solver_params['variant'])+'_lmd_'+str(solver_params['lmd1'])+'_'+str(solver_params['lmd2'])+\
    '_'+solver_params['opt_method']+'_'+solver_params['observed_faces']
basefilename = os.path.join(solver_params['results_folder'], name)

logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)8s:    %(message)s', level=logging.INFO)

logger = logging.getLogger()
logger.info('Starting the problem')
logger.info('Problem description: \n' + '\n'.join([f'{key}: {problem[key]}' for key in problem]))
logger.info('Solver description: \n' + '\n'.join([f'{key}: {solver_params[key]}' for key in solver_params]))

# Create mesh 3D mesh
mesh = fem.BoxMesh(
    fem.Point(problem['lb_x'], problem['lb_y'], problem['lb_z']), 
    fem.Point(problem['rb_x'], problem['rb_y'], problem['rb_z']), 
    solver_params['Nel_x'], solver_params['Nel_y'], solver_params['Nel_z'])
mesh_plot = fem.BoxMesh(
    fem.Point(problem['lb_x'], problem['lb_y'], problem['lb_z']), 
    fem.Point(problem['rb_x'], problem['rb_y'], problem['rb_z']), 
    22, 22, 24)

elements = 'tetrahedron'    

# marker function for observed nodes. Defines logical expression for labeling such nodes.
# Note: use <near> instead of == (equality sign) to avoid problem with numerical rounding.
# Note: use on_boundary marker to be sure only boundary nodes would be considered.
# Note: x[0] defines first coordinate of node x, x[1] - the second, x[2] - the third and so on, i.e., (x,y)=(x[0],x[1])
# Note: to define marker you can use logical operations (e.g., &&, ||) and binary operations (>, <, sqrt, near, etc.)
# Note: the present approach will compile marker function string into efficient C++ code, 
# however it is also possible to write Python class to be a marker function (it will be slower but more epressive). 
# More details with examples can be found here: https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html

boundary_marker = fem.CompiledSubDomain(f'on_boundary')
if solver_params["observed_faces"] == "top":
    observation_marker = fem.CompiledSubDomain(f'near(x[2], {problem["rb_z"]})  && on_boundary')
if solver_params["observed_faces"] == "top_and_sides":
    observation_marker = fem.CompiledSubDomain(f'(near(x[0], {problem["lb_x"]}) || near(x[0], {problem["rb_x"]}) ||'+
                                               f' near(x[1], {problem["lb_y"]}) || near(x[1], {problem["rb_y"]}) ||'+
                                               f' near(x[2], {problem["rb_z"]}))  && on_boundary')

eps = 1e-10 
lbx_marker = problem['lb_x'] + solver_params['bound_num_el']*(problem['rb_x']-problem['lb_x'])/solver_params['Nel_x'] + eps
rbx_marker = problem['rb_x'] - solver_params['bound_num_el']*(problem['rb_x']-problem['lb_x'])/solver_params['Nel_x'] - eps

lby_marker = problem['lb_y'] + solver_params['bound_num_el']*(problem['rb_y']-problem['lb_y'])/solver_params['Nel_y'] + eps
rby_marker = problem['rb_y'] - solver_params['bound_num_el']*(problem['rb_y']-problem['lb_y'])/solver_params['Nel_y'] - eps

lbz_marker = problem['lb_z'] + solver_params['bound_num_el']*(problem['rb_z']-problem['lb_z'])/solver_params['Nel_z'] + eps
rbz_marker = problem['rb_z'] - solver_params['bound_num_el']*(problem['rb_z']-problem['lb_z'])/solver_params['Nel_z'] - eps

near_boundary_marker = fem.CompiledSubDomain(f'(x[0]<{lbx_marker} || x[0]> {rbx_marker}) ||'+
                                             f'(x[1]<{lby_marker} || x[1]> {rby_marker}) ||'+
                                             f'(x[2]<{lbz_marker} || x[2]> {rbz_marker})')


# Define function spaces
P1 = fem.FiniteElement('P', elements, solver_params['degree'])
Q  = fem.FunctionSpace(mesh, P1) 
Q_plot  = fem.FunctionSpace(mesh_plot, P1)

dofs = Q.tabulate_dof_coordinates()
is_dof_on_boundary = np.array([boundary_marker.inside(point, True) for point in dofs])
is_dof_observable = np.array([observation_marker.inside(point, True) for point in dofs])
is_dof_near_boundary = np.array([near_boundary_marker.inside(point, True) for point in dofs])
is_dof_inner = np.array([not near_boundary_marker.inside(point, True) for point in dofs])


#DEFINE SUPPORT OF mu_axf - a polytope in the form [x_l,x_r] x [y_l,y_r] x [z_l, z_r]
if (problem['phantom_lb_x'] < lbx_marker or problem['phantom_rb_x'] > rbx_marker or
    problem['phantom_lb_y'] < lby_marker or problem['phantom_rb_y'] > rby_marker or 
    problem['phantom_lb_z'] < lbz_marker or problem['phantom_rb_z'] > rbz_marker): 
    logger.warning('WARNING: SUPPORT OF mu_axf - [x_l,x_r] x [y_l,y_r] x [z_l, z_r] should be in the complement of near_boundary_marker!')

XiPhantom = fem.Function(Q)

def InsideCube(x,y,z,lb,rb):
    cond_x = (x <= rb[0] and x >= lb[0])
    cond_y = (y <= rb[1] and y >= lb[1])
    cond_z = (z <= rb[2] and z >= lb[2])
        
    return cond_x and cond_y and cond_z

for i in range(dofs.shape[0]): 
    if InsideCube(dofs[i,0],dofs[i,1],dofs[i,2],
                  [problem['phantom_lb_x'],problem['phantom_lb_y'],problem['phantom_lb_z']], 
                  [problem['phantom_rb_x'],problem['phantom_rb_y'],problem['phantom_rb_z']]):
        XiPhantom.vector()[i] = 1
    else:
        XiPhantom.vector()[i] = 0
        
        
# COMPUTE supports for local Total Variation regularization 
TV_cubes_index = construct_TV_cubes(dofs, problem, solver_params)


#### COEFFICIENTS & Bilinear FORMS (weak form) 
# DEFINE coefficents for EXCITATION EQ
# absorption coefficent of the medium at fluorophore excitation wl
mu_axi = fem.Function(Q)
# absoption of Liposyne at 785nm (ICG excitation)
mu_axi.vector()[:] = 0.023 * np.ones(mu_axi.vector()[:].shape)

# absorption coefficent due to fluorophore at fluorophore excitation wl
mu_axf = fem.Function(Q)
#absoption of ICG at 785nm: depends on concetration, approx. 0.5 per 1 micromolar 
ICG_absoption_coeff = fem.Constant(0.5)
# print('TODO: drop ICG_absorption to 0.2')
# support of ICG concentration in the domain 
mu_axf.vector()[:] = ICG_absoption_coeff*XiPhantom.vector()[:]

# scattering coefficent of the medium at fluorophore excitation wl
mu_sxp = fem.Function(Q)
# scattering of Liposyne at 785nm 
mu_sxp.vector()[:] = 9.84 * np.ones(mu_sxp.vector()[:].shape)

# diffusion coefficient 
dx = 1/(3*(mu_axi + mu_axf + mu_sxp))
# print('NOTE: dropped mu_axf from Dx')
# dx = 1/(3*(mu_axi + mu_sxp))

#absoption coefficient 
kx = mu_axi + mu_axf

#Bilinear form for the weak formulation of the EXCITATION EQ 
def ax_form(u,v):
    return dx*fem.dot(fem.grad(u), fem.grad(v))*fem.dx + kx*u*v*fem.dx     + 0.5*g*u*v*fem.ds

#EXCITATION EQ without mu_axf (e.g. no fluorophore ) 
def ax_nomuaxf_form(u,v):
    return dx*fem.dot(fem.grad(u), fem.grad(v))*fem.dx + mu_axi*u*v*fem.dx + 0.5*g*u*v*fem.ds


#EMISSION EQ
# absorption coefficent of the medium at fluorophore emission wl
mu_ami = fem.Function(Q)
# absoption of Liposyne at 830nm (ICG emission) 
mu_ami.vector()[:] = 0.0289 * np.ones(mu_ami.vector()[:].shape)

# absorption coefficent due to fluorophore at fluorophore emission wl 
mu_amf = fem.Function(Q)
# we assume that there is no quenching (Donal:ICG concentration must be below 10-15 micromolar depending on the medium ICG is enclosed in)
mu_amf.vector()[:] = np.zeros(mu_amf.vector()[:].shape)

# scattering coefficent of the medium at fluorophore emission wl 
mu_smp = fem.Function(Q)
# scattering of Liposyne at 830nm 
mu_smp.vector()[:] = 9.84 * np.ones(mu_smp.vector()[:].shape)

# diffusion coefficient 
dm = 1/(3*(mu_ami + mu_amf + mu_smp))
#absoption coefficient 
km = mu_ami + mu_amf

#gamma for the Robin boundary condition comes from Table 1 of this paper for air / Liposyne interface: 
#https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-10-15-653&id=69564 
g = fem.Constant(2.5156)
#ICG quantum efficiency 
Gm = fem.Constant(0.016)

#Bilinear form for the weak formulation of the EMISSION EQ
def am_form(u,v):
    return dm*fem.dot(fem.grad(u), fem.grad(v))*fem.dx + km*u*v*fem.dx + 0.5*g*u*v*fem.ds


### SOURCE term (from real data)
folder = problem['data_folder']
path = folder + problem['source_file']

source_data = np.loadtxt(path)

logger.info('source data norm: '+ str(np.linalg.norm(source_data)))

source_data_medfilt = sps.medfilt(source_data, kernel_size=15)
source_data_gaussfilt = gaussian_filter(source_data_medfilt, sigma=15)
source_data_filt_norm = source_data_gaussfilt / source_data_gaussfilt.max() * 10000

trueSource = source_from_array(Q, source_data_filt_norm, problem)


# Solving the system with real source
# Exitation equation
ux = fem.TrialFunction(Q)
vx = fem.TestFunction(Q)

ax_muax  = ax_form(ux, vx)
rhs = trueSource/2*vx*fem.ds

hatphi = fem.Function(Q)
fem.solve(ax_muax==rhs, hatphi, solver_parameters={'linear_solver': solver_params['linear_solver']})
    
print('hatPhi >= 0:  ', np.all(hatphi.vector()[:]>=0), 'Sum hat_Phi = ', np.sum(hatphi.vector()[:]<0), 'Min hat_Phi = ', np.min(hatphi.vector()[:]))

# emission equation
Sf = fem.Function(Q)
Sf.vector()[:] = np.multiply(mu_axf.vector()[:],hatphi.vector()[:])

rhsSf = float(Gm)*Sf*vx*fem.dx
# am0   = ax_form(ux, vx)
am0   = am_form(ux, vx)

phiSf = fem.Function(Q)
fem.solve(am0==rhsSf, phiSf, solver_parameters={'linear_solver': solver_params['linear_solver']})


### Collecting real observations
# top face observations
folder = problem['data_folder']
path = folder + problem['measurements_top_file']

obs_data = np.loadtxt(path)
obs_data = sps.medfilt(obs_data, kernel_size=15)
obs_data_top = obs_data[100:-100, 120:-95]

obs_data_top_interp = interpolate_observations(obs_data_top)

phi_em_top = TraceXY(phiSf, problem['rb_z'], problem)
phi_em_top_norm = phi_em_top / phi_em_top.max()

obs_data_top_interp_norm = obs_data_top_interp / obs_data_top_interp.max()
obs_diff_top = np.abs(phi_em_top_norm-obs_data_top_interp_norm)
obs_error_top_rel = np.linalg.norm(obs_diff_top)/np.linalg.norm(obs_data_top_interp_norm)
logger.info(f'relative L2 difference on top (no mask) = {obs_error_top_rel}')

mask_top = np.ones_like(obs_data_top_interp_norm)
if solver_params["mask"] == 1:
    mask_top = (obs_data_top_interp_norm>0.9).astype(int)
    
if solver_params["mask"] == 2:
    mask_top = (obs_diff_top<0.1).astype(int)

prod =  obs_data_top_interp_norm/phi_em_top_norm*mask_top

multiplier_top = (obs_data_top_interp/phi_em_top*mask_top).max()
logger.info(f'multiplier_top: {multiplier_top}')

obs_error_top1_rel = np.linalg.norm((phi_em_top_norm-obs_data_top_interp_norm)*mask_top)/np.linalg.norm(obs_data_top_interp_norm*mask_top)
logger.info(f'relative L2 difference on top (mask 1 on) = {obs_error_top1_rel}')

mask_top2 = np.ones_like(obs_data_top_interp_norm)
if solver_params["mask"] == 1:
    mask_top2 = (obs_data_top_interp_norm>0.5).astype(int)
    
if solver_params["mask"] == 2:
    mask_top2 = ((obs_diff_top<0.25) & (phi_em_top_norm>0.2)).astype(int)

prod = obs_data_top_interp_norm/phi_em_top_norm*mask_top2

obs_error_top2_rel = np.linalg.norm((phi_em_top_norm-obs_data_top_interp_norm)*mask_top2)/np.linalg.norm(obs_data_top_interp_norm*mask_top2)
logger.info(f'relative L2 difference on top (mask 2 on) = {obs_error_top2_rel}')


# side face observations
folder = problem['data_folder']
path = folder + problem['measurements_side_file']

obs_data = np.loadtxt(path)
obs_data = sps.medfilt(obs_data, kernel_size=15)
obs_data_side = obs_data[125:-110, 90:-120]

obs_data_side_interp = interpolate_observations(obs_data_side)
obs_data_side_interp = np.flipud(obs_data_side_interp)

phi_em_side = TraceYZ(phiSf, problem['lb_x'], problem)
phi_em_side_norm = phi_em_side / phi_em_side.max()

obs_data_side_interp_norm = obs_data_side_interp / obs_data_side_interp.max()
obs_diff_side = np.abs(phi_em_side_norm-obs_data_side_interp_norm)

logger.info(f'relative L2 difference on side (no mask) = {np.linalg.norm(obs_diff_side)/np.linalg.norm(obs_data_side_interp_norm)}')

mask_side = np.ones_like(obs_data_side_interp_norm)
if solver_params["mask"] == 1:
    mask_side = (obs_data_side_interp_norm>0.9).astype(int)

if solver_params["mask"] == 2:
    logger.warning('if setting obs_diff<0.09, L2 difference error can get to 8%')
    # mask_side = ((obs_diff<0.2) & (phi_em_side_norm>0.5)).astype(int)
    mask_side = ((obs_diff_side<0.09) & (phi_em_side_norm>0.5)).astype(int)

multiplier_side = (obs_data_side_interp/phi_em_side*mask_side).max()
logger.info(f'multiplier_side = {multiplier_side}')

obs_error_side1_rel = np.linalg.norm((phi_em_side_norm-obs_data_side_interp_norm)*mask_side)/np.linalg.norm(obs_data_side_interp_norm*mask_side)
logger.info(f'relative L2 difference on a side (mask 1 on) = {obs_error_side1_rel}')

mask_side2 = np.ones_like(obs_data_side_interp_norm)
if solver_params["mask"] == 1:
    mask_side2 = (obs_data_side_interp_norm>0.5).astype(int)

if solver_params["mask"] == 2:
    mask_side2 = ((obs_diff_side<0.2) & (phi_em_side_norm>0.2)).astype(int)

prod = obs_data_side_interp_norm/phi_em_side_norm*mask_side2
    
obs_error_side2_rel = np.linalg.norm((phi_em_side_norm-obs_data_side_interp_norm)*mask_side2)/np.linalg.norm(obs_data_side_interp_norm*mask_side2)
logger.info(f'relative L2 difference on a side (mask 2 on) = {obs_error_side2_rel}')

### Constructing true observations
# multiplier_top = 1060.54 # current no data mask
if solver_params["mask"] == 0 or solver_params["mask"] == 1:
    multiplier_top = 748.4  # based on comparison with true data
    logger.warning('using precomputed multiplier on top!')
    
# collect observations into a FEM function
y = fem.Function(Q)

int_top = get_interpolant(obs_data_top/multiplier_top)
set_on_plane_z(y, int_top, obs_data_top.shape, problem, z=problem['rb_z'])

if solver_params['observed_faces']=='top_and_sides':
    # side faces over x axis
    int_side = get_interpolant(obs_data_side/multiplier_top)
    set_on_plane_x(y, int_side, obs_data_side.shape, problem, x=0)

    int_side_flipr = get_interpolant(np.fliplr(obs_data_side)/multiplier_top)
    set_on_plane_x(y, int_side_flipr, obs_data_side.shape, problem, x=problem['rb_x'])

    # side faces over y axis
    set_on_plane_y(y, int_side, obs_data_side.shape, problem, y=0)

    set_on_plane_y(y, int_side_flipr, obs_data_side.shape, problem, y=problem['rb_y'])
    
# collect observations into a FEM function
y_mask = fem.Function(Q)

if solver_params["mask"] == 0:
    y_mask.vector()[:] = is_dof_observable.astype(float)
else:
    eps = 0.1
    lvl2 = 0.3
    lvl1 = 1

    # top face
    int_top = get_interpolant(eps*np.ones_like(mask_top)+mask_top2*(lvl2-eps)+mask_top*(lvl1-lvl2))
    set_on_plane_z(y_mask, int_top, mask_top.shape, problem, z=problem['rb_z'])

    # side faces over x axis
    if solver_params['observed_faces']=='top_and_sides':
        side_data = np.flipud(eps*np.ones_like(mask_side)+mask_side2*(lvl2-eps)+mask_side*(lvl1-lvl2))
        int_side = get_interpolant(side_data)
        set_on_plane_x(y_mask, int_side, side_data.shape, problem, x=0)

        int_side_flipr = get_interpolant(np.fliplr(side_data))
        set_on_plane_x(y_mask, int_side_flipr, side_data.shape, problem, x=problem['rb_x'])

        # # side faces over y axis
        set_on_plane_y(y_mask, int_side, side_data.shape, problem, y=0)

        set_on_plane_y(y_mask, int_side_flipr, side_data.shape, problem, y=problem['rb_y'])
        
obs_error_rel = np.linalg.norm((phiSf.vector()[:]-y.vector()[:])*y_mask.vector()[:])/np.linalg.norm((phiSf.vector()[:])*y_mask.vector()[:])
print('relative L2 observation error (mask on) = ', obs_error_rel)


#### OPTIMISATION
# Generating background solutions
start_time = time.time()

ux = fem.TrialFunction(Q)
vx = fem.TestFunction(Q)

hatphi_x = fem.Function(Q)
phibar   = fem.Function(Q)
Sf       = fem.Function(Q)

# Solving excitation equation 
fem.solve(ax_form(ux, vx)==trueSource/2*vx*fem.ds, hatphi_x, 
          solver_parameters={'linear_solver': solver_params['linear_solver']})
print('Phi_x >= 0:  ', np.all(hatphi_x.vector()[:]>=0), '\n#(<0) = ', np.sum(hatphi_x.vector()[:]<0), '2norm % of (<0)', np.linalg.norm(hatphi.vector()[hatphi.vector()[:]<0])/np.linalg.norm(hatphi.vector()[:]), 'Min Phi_x = ', np.min(hatphi.vector()[:]))

fem.solve(ax_nomuaxf_form(ux, vx) == trueSource/2*vx*fem.ds, phibar, 
          solver_parameters={'linear_solver': solver_params['linear_solver']})
print('Phi_bar >= 0: ', np.all(phibar.vector()[:]>=0), '\n#(<0) = ', np.sum(phibar.vector()[:]<0), '2norm % of (<0)', np.linalg.norm(phibar.vector()[phibar.vector()[:]<0])/np.linalg.norm(phibar.vector()[:]),'Min Phi_bar = ', np.min(phibar.vector()[:]))

diff_phibar_x = phibar.vector()[:] - hatphi_x.vector()[:]
print('Phi_bar >= Phi_x', np.all(diff_phibar_x>=0))#, phibar.vector()[diff_phibar_x<0],hatphi_x.vector()[diff_phibar_x<0])
    
if np.min(phibar.vector()[:])<0:
    logger.warning(f'WARNING: FEM solution Phi_bar has {np.sum(phibar.vector()[:]<0)} negative components! Setting negatives to 0!')
    project_to_nonnegative(phibar)

if np.min(hatphi_x.vector()[:])<0:
    logger.warning(f'WARNING: FEM solution Phi_x has {np.sum(hatphi_x.vector()[:]<0)} negative components! Setting negatives to 0!')
    project_to_nonnegative(hatphi_x)


# Solving emission equation
def compute_emission_solution(Sf):
    phi_em = fem.Function(Q)
    fem.solve(am_form(ux, vx)==Sf*vx*float(Gm)*float(ICG_absoption_coeff)*fem.dx, 
              phi_em, solver_parameters={'linear_solver': solver_params['linear_solver']})
    return phi_em

Sf.vector()[:] = np.multiply(XiPhantom.vector()[:], hatphi_x.vector()[:])
hatphi_m = compute_emission_solution(Sf)

err = np.linalg.norm((hatphi_m.vector()[:]-y.vector()[:])*y_mask.vector()[:])/np.linalg.norm(hatphi_m.vector()[:]*y_mask.vector()[:])
if err != obs_error_rel:
    logger.warning('unit test failure, the errors should coincide!')

    
# Compute necessary matrices
S_excit = fem.assemble(ax_form(ux, vx)).array()
f_excit = fem.assemble(rhs)[:]

S_excit_nm = fem.assemble(ax_nomuaxf_form(ux, vx)).array()

S_emit = fem.assemble(am_form(ux, vx)).array()
mM = fem.assemble(ux*vx*fem.dx).array()*float(Gm)*float(ICG_absoption_coeff)

y_vec_tr = y.vector()[is_dof_observable]
y_mask_tr = y_mask.vector()[is_dof_observable]

if solver_params["variant"] == 0:
    logger.info('Inverting emission matrix')
    mF = np.linalg.solve(S_emit,mM)
    mF_tr = mF[is_dof_observable,:]
    
    
logger.info(f'====================== Initial iteration of {solver_params["name"]} method starting. Step 1')
if solver_params["name"] == "born":
    Phi_x_var = cp.Parameter(shape=Q.dim(), nonneg=True)
    Phi_x_var.value = phibar.vector()[:]
    
if solver_params["name"] == "hybrid":
    Svar0 = cp.Parameter(shape=Q.dim(), nonneg=True)
    Svar0.value = np.zeros(Q.dim())
    # Svar0.value = Svar1.value

    Phi_x_var  = cp.Variable(shape=Q.dim(), nonneg=True)
    Phi_m_var0 = cp.Variable(shape=Q.dim(), nonneg=True)

    misfit_norm = cp.Parameter(nonneg=True)
    misfit_norm.value = 1/np.linalg.norm(y_vec_tr*y_mask_tr,2)**2

    Cost1 = misfit_norm*cp.sum_squares(cp.multiply(Phi_m_var0[is_dof_observable] - y_vec_tr, y_mask_tr)) \
          + solver_params['lmd1']*cp.sum_squares(S_excit_nm@Phi_x_var+mM@(cp.multiply(Phi_x_var, Svar0))/float(Gm)-f_excit) \
          + solver_params['lmd2']*cp.sum_squares(S_emit@Phi_m_var0-mM@(cp.multiply(Phi_x_var, Svar0)))

    prob_step1 = cp.Problem(cp.Minimize(Cost1))
    if solver_params["opt_method"] == cp.OSQP:
        prob_step1.solve(cp.OSQP, verbose = True, max_iter=solver_params['max_iter'], 
                         ignore_dpp = True, eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
    elif solver_params["opt_method"] == cp.SCS:
        prob_step1.solve(cp.SCS, verbose = True, max_iters=solver_params['max_iter'], 
                         ignore_dpp = True, eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
    elif solver_params["opt_method"] == cp.ECOS:
        prob_step1.solve(cp.ECOS, verbose = True, max_iters=solver_params['max_iter'], 
                         ignore_dpp = True)
    elif solver_params["opt_method"] == cp.MOSEK:
        prob_step1.solve(cp.MOSEK, verbose = True, ignore_dpp = True, 
                   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.free, 
                                 mosek.dparam.intpnt_co_tol_pfeas: 1e-8, mosek.dparam.intpnt_co_tol_infeas: 1e-12})

    temp_f = fem.Function(Q)
    temp_f.vector()[:] = Phi_m_var0.value
    phi_m0_error_cont = np.sqrt(fem.assemble((hatphi_m-temp_f)**2*fem.dx)/fem.assemble((hatphi_m)**2*fem.dx))
    phi_m0_error_disc = np.linalg.norm(hatphi_m.vector()[:]-Phi_m_var0.value)/np.linalg.norm(hatphi_m.vector()[:])

    logger.info('   Initial iteration info (after step 1):\n'
        +f'   misfit    : {misfit_norm.value*cp.sum_squares(cp.multiply(Phi_m_var0[is_dof_observable] - y_vec_tr, y_mask_tr)).value}\n'
        +f'   excitation: {solver_params["lmd1"]*cp.sum_squares(S_excit_nm@Phi_x_var+mM@(cp.multiply(Phi_x_var, Svar0))/float(Gm)-f_excit).value}\n'
        +f'   emission  : {solver_params["lmd2"]*cp.sum_squares(S_emit@Phi_m_var0-mM@(cp.multiply(Phi_x_var, Svar0))).value}\n'
        +f'   number iters: {prob_step1.solver_stats.num_iters}\n'
        +f'   solve 1 time: {prob_step1.solver_stats.solve_time}\n'
        +f'   phi_m   reconstruction error:  continuous: {phi_m0_error_cont};  discrete: {phi_m0_error_disc}\n')
time_lapse1 = time.time() - start_time


logger.info(f'Initial iteration. Step 2. Solver variant: {solver_params["variant"]}')
start_time = time.time()

Phi_x_var0 = cp.Parameter(shape=phibar.vector()[:].shape[0], nonneg=True)
Phi_x_var0.value = Phi_x_var.value
if solver_params["init_test"]:
    logger.info(f'This is initial step test only.')
    Phi_x_var0.value = hatphi_x.vector()[:]

Svar1     = cp.Variable(shape=Q.dim(), nonneg=True)
Phi_m_var = cp.Variable(shape=Q.dim(), nonneg=True)

norm_bound = cp.Parameter(nonneg=True)
norm_bound.value = 1
EmissionEq_tol = cp.Parameter(nonneg=True)
# TODO: making it lower then 1e-4 results in worse reconstruction result and eventual failure
EmissionEq_tol.value   = 1e-4

tv_norm = cp.Parameter(nonneg=True)
tv_norm.value = 1/TV_cubes_index.shape[0]
tv_reg = compute_tv(TV_cubes_index, Svar1)

misfit_norm = cp.Parameter(nonneg=True)
misfit_norm.value = 1/np.linalg.norm(y_vec_tr*y_mask_tr,2)**2


# ALGORITHM BORN, variant 1
# provides 26% error
if solver_params["variant"] == 0:
    Cost2 = misfit_norm*cp.sum_squares(cp.multiply(mF_tr@cp.multiply(Phi_x_var0,Svar1) - y_vec_tr, y_mask_tr)) \
          + tv_norm*tv_reg
    Constraints = [Svar1<=1, Svar1[is_dof_near_boundary]==0]


# ALGORITHM BORN, variant 2
# cp.sum(Svar1)>=200 - such regularisation is really needed, otherwise we get much worse results
if solver_params["variant"] == 1:
    Cost2 = misfit_norm*cp.sum_squares(cp.multiply(Phi_m_var[is_dof_observable] - y_vec_tr, y_mask_tr)) \
          + tv_norm*tv_reg
    Constraints = [Svar1<=1, 
                   cp.sum(Svar1)>=200, 
                   Svar1[is_dof_near_boundary]==0,
                   cp.sum(S_emit@Phi_m_var-mM@(cp.multiply(Phi_x_var0,Svar1)))==0]    

# ALGORITHM BORN, variant 3
# provides 97% error after 7875 iterations
if solver_params["variant"] == 2:
    Cost2 = misfit_norm*cp.sum_squares(cp.multiply(Phi_m_var[is_dof_observable] - y_vec_tr, y_mask_tr)) \
          + cp.sum_squares(S_emit@Phi_m_var-mM@(cp.multiply(Phi_x_var0,Svar1))) \
          + tv_norm*tv_reg
    Constraints = [Svar1<=1, 
                   # tv_norm*tv_reg <= 0.01, 
                   cp.sum(Svar1)>=200, 
                   Svar1[is_dof_near_boundary]==0]


# ALGORITHM BORN, variant 4
# provides 99% error after small amount of iterations
if solver_params["variant"] == 3:
    Cost2 = misfit_norm*cp.sum_squares(cp.multiply(Phi_m_var[is_dof_observable] - y_vec_tr, y_mask_tr)) \
          + tv_norm*tv_reg
    Constraints = [Svar1<=1, 
                   # tv_norm*tv_reg <= 0.01, 
                   cp.sum(Svar1)>=200, 
                   cp.sum(S_emit@Phi_m_var-mM@(cp.multiply(Phi_x_var0,Svar1))) <= EmissionEq_tol,
                   Svar1[is_dof_near_boundary]==0]

prob_step2 = cp.Problem(cp.Minimize(Cost2), Constraints)
if solver_params["opt_method"] == cp.OSQP:
    prob_step2.solve(cp.OSQP, verbose = True, max_iter=solver_params['max_iter'], ignore_dpp = True,
                     eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
elif solver_params["opt_method"] == cp.SCS:
    prob_step2.solve(cp.SCS, verbose = True, max_iters=solver_params['max_iter'], ignore_dpp = True,
                     eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
elif solver_params["opt_method"] == cp.ECOS:
    prob_step2.solve(cp.ECOS, verbose = True, max_iters=solver_params['max_iter'], ignore_dpp = True)
elif solver_params["opt_method"] == cp.MOSEK:
    prob_step2.solve(cp.MOSEK, verbose = True, ignore_dpp = True, 
                     mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.free, 
                                   mosek.dparam.intpnt_co_tol_pfeas: 1e-8, mosek.dparam.intpnt_co_tol_infeas: 1e-12})

time_lapse2 = time.time() - start_time
logger.info('Initial iteration. Step 2 is finished.')


# Optimisation algorithm (step 2) status summary
gk    = cp.Parameter(len(XiPhantom.vector()[:]))
gk.value = XiPhantom.vector()[:]

tv_gk = compute_tv(TV_cubes_index, gk)

mu_axf_est = fem.Function(Q)
mu_axf_est.vector()[:] = np.multiply(Phi_x_var0.value, gk.value)
phi_em_est = compute_emission_solution(mu_axf_est).vector()[:]

if solver_params["variant"] == 0:
    dmsft      = (misfit_norm*cp.sum_squares(cp.multiply(mF_tr@cp.multiply(Phi_x_var0,Svar1) - y_vec_tr, y_mask_tr))).value
    dmsft_true = (misfit_norm*cp.sum_squares(cp.multiply(mF_tr@cp.multiply(Phi_x_var0,gk) - y_vec_tr, y_mask_tr))).value
    cost_true  = dmsft_true + (tv_norm*tv_reg).value
    
    Phi_m_var  = cp.Parameter(shape=Q.dim())
    Phi_m_var.value = mF@np.multiply(Phi_x_var0.value,Svar1.value)
else:
    dmsft      = (misfit_norm*cp.sum_squares(cp.multiply(Phi_m_var[is_dof_observable] - y_vec_tr, y_mask_tr))).value
    dmsft_true = (misfit_norm*cp.sum_squares(cp.multiply(phi_em_est[is_dof_observable] - y_vec_tr, y_mask_tr))).value
    
    if solver_params["variant"] == 1:
        cost_true = dmsft_true
    if solver_params["variant"] == 2:
        cost_true = dmsft_true + (tv_norm*tv_reg).value + cp.sum_squares(S_emit@Phi_m_var-mM@(cp.multiply(Phi_x_var0,gk))).value
    if solver_params["variant"] == 3:
        cost_true = dmsft_true + (tv_norm*tv_reg).value

sol1 = fem.Function(Q)
sol1.vector()[:] = Svar1.value
recon_error = np.sqrt(fem.assemble((XiPhantom-sol1)**2*fem.dx)/fem.assemble((XiPhantom)**2*fem.dx))
l2_adj_error = l2_adj_metric(Svar1.value, XiPhantom.vector()[:], Q)
dice = dice_metric(Svar1.value, XiPhantom.vector()[:])

message = '   Initial iteration summary:\n' +\
        f'   Reconstruction error = {recon_error}\n' +\
        f'   adjusted recon error = {l2_adj_error}\n' +\
        f'   dice = {dice}\n' +\
        f'   Data misfit = {dmsft}   TV = {(tv_norm*tv_reg).value}\n' +\
        f'   True misfit = {dmsft_true}   TV = {(tv_norm*tv_gk).value}\n' +\
        f'   Cost = {Cost2.value}   True cost = {dmsft_true + (tv_norm*tv_reg).value}\n' +\
        f'   Non-zero component = {(Svar1.value>0.1).sum()}   True non-zero component = {(gk.value>0.1).sum()}\n'
if solver_params["variant"] != 0:
    message = message + f'\n   Emission equation constraint = {cp.sum_squares(S_emit@Phi_m_var-mM@(cp.multiply(Phi_x_var0,Svar1))).value}'

logger.info(message)


# Initial iteration info
temp_f = fem.Function(Q)
    
temp_f.vector()[:] = Svar1.value
recon_error_cont = np.sqrt(fem.assemble((XiPhantom-temp_f)**2*fem.dx)/fem.assemble((XiPhantom)**2*fem.dx))
recon_error_disc = np.linalg.norm(XiPhantom.vector()[:]-Svar1.value)/np.linalg.norm(XiPhantom.vector()[:])

temp_f.vector()[:] = Phi_x_var0.value
phi_x_error_cont = np.sqrt(fem.assemble((hatphi_x-temp_f)**2*fem.dx)/fem.assemble((hatphi_x)**2*fem.dx))
phi_x_error_disc = np.linalg.norm(hatphi_x.vector()[:]-Phi_x_var0.value)/np.linalg.norm(hatphi_x.vector()[:])

temp_f.vector()[:] = Phi_m_var.value
phi_m_error_cont = np.sqrt(fem.assemble((hatphi_m-temp_f)**2*fem.dx)/fem.assemble((hatphi_m)**2*fem.dx))
phi_m_error_disc = np.linalg.norm(hatphi_m.vector()[:]-Phi_m_var.value)/np.linalg.norm(hatphi_m.vector()[:])

logger.info('   Initial iteration info (after step 2):\n'
           +f'   number 2 iters: {prob_step2.solver_stats.num_iters}\n'
           +f'   solve 2 time: {prob_step2.solver_stats.solve_time}\n'
           +f'   step 1 time: {time_lapse1}\n'
           +f'   step 2 time: {time_lapse2}\n'
           +f'   total  time: {time_lapse1+time_lapse2}\n'
           +f'   phi_x_est norm: {np.linalg.norm(Phi_x_var0.value)};  hatphi_x  norm: {np.linalg.norm(hatphi_x.vector()[:])}\n'
           +f'   phi_m_est norm: {np.linalg.norm(Phi_m_var.value)};  hatphi_m  norm: {np.linalg.norm(hatphi_m.vector()[:])}\n'
           +f'   phantom reconstruction error:  continuous: {recon_error_cont};  discrete: {recon_error_disc}\n'
           +f'   phi_x   reconstruction error:  continuous: {phi_x_error_cont};  discrete: {phi_x_error_disc}\n'
           +f'   phi_m   reconstruction error:  continuous: {phi_m_error_cont};  discrete: {phi_m_error_disc}\n')


if not solver_params["init_test"]:
    # Storing iterative information
    run_info = {
        'solve 1 time': [],
        'solve 2 time': [prob_step2.solver_stats.solve_time],
        'number 1 iters': [],
        'number 2 iters': [prob_step2.solver_stats.num_iters],
        'step 1 time': [time_lapse1],
        'step 2 time': [time_lapse2],
        'total time': [time_lapse1+time_lapse2],
        'phant. recon. error': [recon_error_cont],
        'phi_x  recon. error': [phi_x_error_cont],
        'phi_m  recon. error': [phi_m_error_cont],
        'phi_m0 recon. error': [],
        'phantom_diff': [],
        'adjusted l2 with true': [l2_adj_error],
        'adjusted l2 with prev': [],    
        'dice with true': [dice],
        'dice with prev': []
    }

    if solver_params["name"] == "hybrid":
        run_info['solve 1 time'].append(prob_step1.solver_stats.solve_time)
        run_info['number 1 iters'].append(prob_step1.solver_stats.num_iters)
        run_info['phi_m0 recon. error'].append(phi_m0_error_cont)

    XiPhantom_est_prev = Svar1.value.copy()
    phi_x_prev = Phi_x_var0.value.copy()
    phi_m_prev = Phi_m_var.value.copy()

    np.save(basefilename+f'_phantom_{0}', Svar1.value)
    np.save(basefilename+f'_phi_x_est_{0}', Phi_x_var0.value)
    np.save(basefilename+f'_phi_m_est_{0}', Phi_m_var.value)

    phantom_diff = 1
    itr = 1

    while (phantom_diff > solver_params["step_diff"]) and (itr <= solver_params["max_steps"]):
        logger.info(f'====================== iteration: {itr} starting')
        start_time = time.time()

        mu_axf.vector()[:] = ICG_absoption_coeff*Svar1.value
        # Step 1:
        if solver_params["name"] == "born":
            phi_x_est = fem.Function(Q)
            fem.solve(ax_muax==rhs, phi_x_est, solver_parameters={'linear_solver': solver_params['linear_solver']})
            project_to_nonnegative(phi_x_est)
            Phi_x_var0.value = phi_x_est.vector()[:]

        if solver_params["name"] == "hybrid":
            Svar0.value = Svar1.value

            # need to update S_excit_nm after each update of chi (Svar)
            S_excit_nm = fem.assemble(ax_nomuaxf_form(ux, vx)).array()
            Cost1 = misfit_norm*cp.sum_squares(cp.multiply(Phi_m_var0[is_dof_observable] - y_vec_tr, y_mask_tr)) \
                  + solver_params['lmd1']*cp.sum_squares(S_excit_nm@Phi_x_var+mM@(cp.multiply(Phi_x_var, Svar0))/float(Gm)-f_excit) \
                  + solver_params['lmd2']*cp.sum_squares(S_emit@Phi_m_var0-mM@(cp.multiply(Phi_x_var, Svar0)))
            prob_step1 = cp.Problem(cp.Minimize(Cost1))

            if solver_params["opt_method"] == cp.OSQP:
                prob_step1.solve(cp.OSQP, verbose = False, max_iter=solver_params['max_iter'], 
                                 ignore_dpp = True, eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
            elif solver_params["opt_method"] == cp.SCS:
                prob_step1.solve(cp.SCS, verbose = False, max_iters=solver_params['max_iter'], 
                                 ignore_dpp = True, eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
            elif solver_params["opt_method"] == cp.ECOS:
                prob_step1.solve(cp.ECOS, verbose = False, max_iters=solver_params['max_iter'], 
                                 ignore_dpp = True)
            elif solver_params["opt_method"] == cp.MOSEK:
                prob_step1.solve(cp.MOSEK, ignore_dpp = True, 
                           mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.free, 
                                         mosek.dparam.intpnt_co_tol_pfeas: 1e-8, mosek.dparam.intpnt_co_tol_infeas: 1e-12})

            Phi_x_var0.value = Phi_x_var.value

            temp_f.vector()[:] = Phi_m_var0.value
            phi_m0_error_cont = np.sqrt(fem.assemble((hatphi_m-temp_f)**2*fem.dx)/fem.assemble((hatphi_m)**2*fem.dx))
            phi_m0_error_disc = np.linalg.norm(hatphi_m.vector()[:]-Phi_m_var0.value)/np.linalg.norm(hatphi_m.vector()[:])        
            logger.info('after step 1:\n'
                +f'   misfit    : {misfit_norm.value*cp.sum_squares(cp.multiply(Phi_m_var0[is_dof_observable] - y_vec_tr, y_mask_tr)).value}\n'
                +f'   excitation: {solver_params["lmd1"]*cp.sum_squares(S_excit_nm@Phi_x_var+mM@(cp.multiply(Phi_x_var, Svar0))/float(Gm)-f_excit).value}\n'
                +f'   emission  : {solver_params["lmd2"]*cp.sum_squares(S_emit@Phi_m_var0-mM@(cp.multiply(Phi_x_var, Svar0))).value}\n'
                +f'   number iters: {prob_step1.solver_stats.num_iters}\n'
                +f'   solve 1 time: {prob_step2.solver_stats.solve_time}\n'
                +f'   phi_m   reconstruction error:  continuous: {phi_m0_error_cont};  discrete: {phi_m0_error_disc}\n')
        time_lapse1 = time.time() - start_time

        # Step 2:
        start_time = time.time()
        if solver_params["opt_method"] == cp.OSQP or solver_params["opt_method"] == cp.ECOS:
            prob_step2.solve(cp.OSQP, max_iter=solver_params['max_iter'], ignore_dpp = True, 
                             warm_start=False, verbose = False,
                             eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
        elif solver_params["opt_method"] == cp.SCS:
            prob_step2.solve(solver_params["opt_method"], max_iters=solver_params['max_iter'], ignore_dpp = True, 
                             warm_start=False, verbose = False,
                             eps_abs=solver_params["eps_abs"], eps_rel=solver_params["eps_rel"])
        elif solver_params["opt_method"] == cp.ECOS:
            prob_step2.solve(cp.ECOS, max_iters=solver_params['max_iter'], ignore_dpp = True,
                             warm_start=False, verbose = False)
        elif solver_params["opt_method"] == cp.MOSEK:
            prob_step2.solve(cp.MOSEK, ignore_dpp = True, 
                             mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.free, 
                                           mosek.dparam.intpnt_co_tol_pfeas: 1e-8, mosek.dparam.intpnt_co_tol_infeas: 1e-12})        
        if solver_params["variant"] == 0:
            Phi_m_var.value = mF@np.multiply(Phi_x_var0.value, Svar1.value)
        time_lapse2 = time.time() - start_time

        # Svar1.value[Svar1.value>=0.5] = 1
        # Svar1.value[Svar1.value<0.5] = 0

        temp_f.vector()[:] = Svar1.value
        recon_error_cont = np.sqrt(fem.assemble((XiPhantom-temp_f)**2*fem.dx)/fem.assemble((XiPhantom)**2*fem.dx))
        recon_error_disc = np.linalg.norm(XiPhantom.vector()[:]-Svar1.value)/np.linalg.norm(XiPhantom.vector()[:])

        temp_f.vector()[:] = Phi_x_var0.value
        phi_x_error_cont = np.sqrt(fem.assemble((hatphi_x-temp_f)**2*fem.dx)/fem.assemble((hatphi_x)**2*fem.dx))
        phi_x_error_disc = np.linalg.norm(hatphi_x.vector()[:]-Phi_x_var0.value)/np.linalg.norm(hatphi_x.vector()[:])

        temp_f.vector()[:] = Phi_m_var.value
        phi_m_error_cont = np.sqrt(fem.assemble((hatphi_m-temp_f)**2*fem.dx)/fem.assemble((hatphi_m)**2*fem.dx))
        phi_m_error_disc = np.linalg.norm(hatphi_m.vector()[:]-Phi_m_var.value)/np.linalg.norm(hatphi_m.vector()[:])

        phantom_diff = np.linalg.norm(XiPhantom_est_prev - Svar1.value)/np.linalg.norm(XiPhantom_est_prev)
        phi_x_diff   = np.linalg.norm(phi_x_prev - Phi_x_var0.value)/np.linalg.norm(phi_x_prev)
        phi_m_diff   = np.linalg.norm(phi_m_prev - Phi_m_var.value)/np.linalg.norm(phi_m_prev)

        dice = dice_metric(Svar1.value, XiPhantom.vector()[:])
        dice_diff = dice_metric(Svar1.value, XiPhantom_est_prev)

        l2_adj_error = l2_adj_metric(Svar1.value, XiPhantom.vector()[:], Q)
        l2_adj_diff  = l2_adj_metric(Svar1.value, XiPhantom_est_prev, Q)

        logger.info('   iteration info after step 2:\n'
                   +f'   number 2 iters: {prob_step2.solver_stats.num_iters}\n'
                   +f'   solve 2 time: {prob_step2.solver_stats.solve_time}\n'
                   +f'   step 1 time: {time_lapse1}\n'
                   +f'   step 2 time: {time_lapse2}\n'
                   +f'   total  time: {time_lapse1+time_lapse2}\n'
                   +f'   phantom reconstruction error:  continuous: {recon_error_cont};  discrete: {recon_error_disc}\n'
                   +f'   phi_x   reconstruction error:  continuous: {phi_x_error_cont};  discrete: {phi_x_error_disc}\n'
                   +f'   phi_m   reconstruction error:  continuous: {phi_m_error_cont};  discrete: {phi_m_error_disc}\n'                
                   +f'   per iteration phantom difference:  {phantom_diff}\n'
                   +f'   per iteration  phi_x  difference: {phi_x_diff}\n'
                   +f'   per iteration  phi_m  difference: {phi_m_diff}\n'
                   +f'   adjusted l2 with true: {l2_adj_error}\n'
                   +f'   adjusted l2 with prev: {l2_adj_diff}\n'
                   +f'   dice with true: {dice}\n'
                   +f'   dice with prev: {dice_diff}\n')

        run_info['solve 2 time'].append(prob_step2.solver_stats.solve_time)
        run_info['number 2 iters'].append(prob_step2.solver_stats.num_iters)
        run_info['step 1 time'].append(time_lapse1)
        run_info['step 2 time'].append(time_lapse2)
        run_info['total time'].append(time_lapse1+time_lapse2)
        run_info['phant. recon. error'].append(recon_error_cont)
        run_info['phi_x  recon. error'].append(phi_x_error_cont)
        run_info['phi_m  recon. error'].append(phi_m_error_cont)
        run_info['phantom_diff'].append(phantom_diff)
        run_info['adjusted l2 with true'].append(l2_adj_error)
        run_info['adjusted l2 with prev'].append(l2_adj_diff)
        run_info['dice with true'].append(dice)
        run_info['dice with prev'].append(dice_diff)

        if solver_params["name"] == "hybrid":
            run_info['solve 1 time'].append(prob_step1.solver_stats.solve_time)
            run_info['number 1 iters'].append(prob_step1.solver_stats.num_iters)
            run_info['phi_m0 recon. error'].append(phi_m0_error_cont)

        np.save(basefilename+f'_phantom_{itr}', Svar1.value)
        np.save(basefilename+f'_phi_x_est_{itr}', Phi_x_var0.value)
        np.save(basefilename+f'_phi_m_est_{itr}', Phi_m_var.value)

        XiPhantom_est_prev = Svar1.value.copy()
        phi_x_prev = Phi_x_var0.value.copy()
        phi_m_prev = Phi_m_var.value.copy()

        itr = itr+1

    logger.info('Iterative algorithm has finished')


    logger.info('Iterations summary:\n'
               + f'   solver iters  : {itr}\n'
               + f'   solve 1 times : {run_info["solve 1 time"]}\n'
               + f'   solve 2 times : {run_info["solve 2 time"]}\n'
               + f'   step 1 time   : {run_info["step 1 time"]}\n'
               + f'   step 2 time   : {run_info["step 2 time"]}\n'
               + f'   total times   : {run_info["total time"]}\n'
               + f'   total execution time   : {sum(run_info["total time"])}\n'
               + f'   execution time per iter: {sum(run_info["total time"])/itr}\n'
               + f'   number 1 iters: {run_info["number 1 iters"]}\n'
               + f'   number 2 iters: {run_info["number 2 iters"]}\n'
               + f'   phant. recon. error: {run_info["phant. recon. error"]}\n'
               + f'   phi_x  recon. error: {run_info["phi_x  recon. error"]}\n'
               + f'   phi_m  recon. error: {run_info["phi_m  recon. error"]}\n'
               + f'   phi_m0 recon. error: {run_info["phi_m0 recon. error"]}\n'
               + f'   phantom diff. : {run_info["phantom_diff"]}\n'
               + f'   adjusted l2 with true: {run_info["adjusted l2 with true"]}\n'
               + f'   adjusted l2 with prev: {run_info["adjusted l2 with prev"]}\n'
               + f'   dice with true : {run_info["dice with true"]}\n'
               + f'   dice with prev : {run_info["dice with prev"]}\n')
    
    
    # Optimisation algorithm (step 2) final status summary
    gk    = cp.Parameter(len(XiPhantom.vector()[:]))
    gk.value = XiPhantom.vector()[:]

    tv_gk = compute_tv(TV_cubes_index, gk)

    mu_axf_est = fem.Function(Q)
    mu_axf_est.vector()[:] = np.multiply(Phi_x_var0.value, gk.value)
    phi_em_est = compute_emission_solution(mu_axf_est).vector()[:]

    if solver_params["variant"] == 0:
        dmsft      = (misfit_norm*cp.sum_squares(cp.multiply(mF_tr@cp.multiply(Phi_x_var0,Svar1) - y_vec_tr, y_mask_tr))).value
        dmsft_true = (misfit_norm*cp.sum_squares(cp.multiply(mF_tr@cp.multiply(Phi_x_var0,gk) - y_vec_tr, y_mask_tr))).value
        cost_true  = dmsft_true + (tv_norm*tv_reg).value
    else:
        dmsft      = (misfit_norm*cp.sum_squares(cp.multiply(Phi_m_var[is_dof_observable] - y_vec_tr, y_mask_tr))).value
        dmsft_true = (misfit_norm*cp.sum_squares(cp.multiply(phi_em_est[is_dof_observable] - y_vec_tr, y_mask_tr))).value

        if solver_params["variant"] == 1:
            cost_true = dmsft_true
        if solver_params["variant"] == 2:
            cost_true = dmsft_true + (tv_norm*tv_reg).value + cp.sum_squares(S_emit@Phi_m_var-mM@(cp.multiply(Phi_x_var0,gk))).value
        if solver_params["variant"] == 3:
            cost_true = dmsft_true + (tv_norm*tv_reg).value

    sol1 = fem.Function(Q)
    sol1.vector()[:] = Svar1.value        
    recon_error = np.sqrt(fem.assemble((XiPhantom-sol1)**2*fem.dx)/fem.assemble((XiPhantom)**2*fem.dx))
    message = '   Final iteration summary:\n' +\
            f'   Reconstruction error = {recon_error}\n' +\
            f'   Data misfit = {dmsft}   TV = {(tv_norm*tv_reg).value}\n' +\
            f'   True misfit = {dmsft_true}   TV = {(tv_norm*tv_gk).value}\n' +\
            f'   Cost = {Cost2.value}   True cost = {dmsft_true + (tv_norm*tv_reg).value}\n' +\
            f'   Non-zero component = {(Svar1.value>0.1).sum()}   True non-zero component = {(gk.value>0.1).sum()}'
    if solver_params["variant"] != 0:
        message = message + f'\n   Emission equation constraint = {cp.sum_squares(S_emit@Phi_m_var-mM@(cp.multiply(Phi_x_var0,Svar1))).value}'

    logger.info(message)

    logger.info('the END')
