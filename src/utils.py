import numpy as np
import matplotlib.pyplot as plt
import fenics as fem
import scipy.interpolate as spi
import cvxpy as cp


res = 100

def TraceXY(func, z, problem):
    xx = np.arange(problem['lb_x'],problem['rb_x'],(problem['rb_x']-problem['lb_x'])/res)
    yy = np.arange(problem['lb_y'],problem['rb_y'],(problem['rb_y']-problem['lb_y'])/res)
    val2D = np.zeros((xx.shape[0],yy.shape[0]))
    for i in range(xx.shape[0]): 
        for j in range(yy.shape[0]): 
            val2D[i,j] = func(fem.Point(xx[j],yy[i],z))
                    
    return val2D

def TraceYZ(func, x, problem):
    zz = np.arange(problem['lb_z'],problem['rb_z'],(problem['rb_z']-problem['lb_z'])/res)
    yy = np.arange(problem['lb_y'],problem['rb_y'],(problem['rb_y']-problem['lb_y'])/res)
    val2D = np.zeros((zz.shape[0], yy.shape[0]))
    for i in range(zz.shape[0]): 
        for j in range(yy.shape[0]): 
            val2D[i,j] = func(fem.Point(x, yy[j], zz[i]))
                    
    return val2D

def TraceXZ(func, y):
    zz = np.arange(problem['lb_z'],problem['rb_z'],(problem['rb_z']-problem['lb_z'])/res)
    xx = np.arange(problem['lb_x'],problem['rb_x'],(problem['rb_x']-problem['lb_x'])/res)
    val2D = np.zeros((xx.shape[0], zz.shape[0]))
    for i in range(xx.shape[0]): 
        for j in range(zz.shape[0]): 
            val2D[i,j] = func(fem.Point(xx[j], y, zz[i]))
                    
    return val2D

def plot_trace(func, z):
    trace = Trace(func, z)
    img = plt.imshow(trace)
    return img

def plot_traceYZ(func, x):
    trace = TraceYZ(func, x)
    img = plt.imshow(trace)
    return img

def plot_traceXZ(func, y):
    trace = TraceXZ(func, y)
    img = plt.imshow(trace)
    return img

def exponential_source_on_top(Q, problem):
    dofs = Q.tabulate_dof_coordinates()
    is_dof_on_boundary = np.array([fem.near(problem['rb_z'], point[2]) for point in dofs])
    
    trueSource = fem.Function(Q)
    value = 1e4*np.exp(-(dofs[:,0]-(problem['rb_x']+problem['lb_x'])/2)**2/0.35 - (dofs[:,1] - (problem['rb_y']+problem['lb_y'])/2)**2/0.35)
    trueSource.vector()[:] = value * is_dof_on_boundary
    
    return trueSource

def source_from_array(Q, source_data, problem):
    # convert source data to the Fenics functin
    trueSource = fem.Function(Q)
    dofs = Q.tabulate_dof_coordinates()
    
    for i in range(dofs.shape[0]):
        if fem.near(dofs[i,2], problem['rb_z']):
            p_x = round((dofs[i,0] / problem['rb_x']) * (source_data.shape[1]-1))
            p_y = round((dofs[i,1] / problem['rb_y']) * (source_data.shape[0]-1))

            trueSource.vector()[i] = source_data[p_y, p_x]
    return trueSource


def construct_TV_cubes(dofs, problem, solver_params):
    len_x = (problem['rb_x'] - problem['lb_x']) / solver_params['Nc_x']
    len_y = (problem['rb_y'] - problem['lb_y']) / solver_params['Nc_y']
    len_z = (problem['rb_z'] - problem['lb_z']) / solver_params['Nc_z']

    TV_cubes_index = np.zeros(shape=[solver_params['Nc_x']*solver_params['Nc_y']*solver_params['Nc_z'], dofs.shape[0]], dtype=int)

    for ii in range(dofs.shape[0]):
        jx = np.min([np.floor(dofs[ii,0] / len_x), solver_params['Nc_x']-1])
        jy = np.min([np.floor(dofs[ii,1] / len_y), solver_params['Nc_y']-1])
        jz = np.min([np.floor(dofs[ii,2] / len_z), solver_params['Nc_z']-1])

        cube_ind = int(jx*solver_params['Nc_y']*solver_params['Nc_z'] + jy*solver_params['Nc_z'] + jz)

        TV_cubes_index[cube_ind, ii] = 1
    return TV_cubes_index


def interpolate_observations(obs_data):
    x = np.arange(0, res+0.0001, res/(obs_data.shape[1]-1))
    y = np.arange(0, res+0.0001, res/(obs_data.shape[0]-1))
    xx, yy = np.meshgrid(x, y)

    x_new = np.arange(0, res)
    y_new = np.arange(0, res)

    f = spi.interp2d(x, y, obs_data)
    obs_data_interp = f(x_new, y_new)
    
    return obs_data_interp

def get_interpolant(obs_data):
    x = np.arange(0, obs_data.shape[1], 1)
    y = np.arange(0, obs_data.shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    return spi.interp2d(x, y, obs_data)

def set_on_plane_z(f_fem, f_data, shape, problem, z=0):
    dofs = f_fem.function_space().tabulate_dof_coordinates()
    for i in range(dofs.shape[0]):
        if fem.near(z, dofs[i,2]):
            f_fem.vector()[i] = f_data(dofs[i,0]/problem['rb_x']*shape[1], dofs[i,1]/problem['rb_y']*shape[0])[0]
            # print(dofs[i,0]/rb_x*shape[1], dofs[i,1]/rb_y*shape[0])
            
def set_on_plane_x(f_fem, f_data, shape, problem, x=0):
    dofs = f_fem.function_space().tabulate_dof_coordinates()
    for i in range(dofs.shape[0]):
        if fem.near(x, dofs[i,0]):
            # need to flip values over z-axis
            f_fem.vector()[i] = f_data(dofs[i,1]/problem['rb_y']*shape[1], shape[0]-dofs[i,2]/problem['rb_z']*shape[0])
            
def set_on_plane_y(f_fem, f_data, shape, problem, y=0):
    dofs = f_fem.function_space().tabulate_dof_coordinates()
    for i in range(dofs.shape[0]):
        if fem.near(y, dofs[i,1]):
            # need to flip values over z-axis
            f_fem.vector()[i] = f_data(dofs[i,0]/problem['rb_x']*shape[1], shape[0]-dofs[i,2]/problem['rb_z']*shape[0])
            
def compute_tv(tv_cubes_index, Svar1):
    TVreg = cp.tv(Svar1[tv_cubes_index[0,:]])
    for jj in range(1,tv_cubes_index.shape[0]):
        node_reg = 1/tv_cubes_index[jj,:].sum()**(2/3)
        TVreg = TVreg + node_reg*cp.tv(Svar1[tv_cubes_index[jj,:].astype(bool)])
    return TVreg            

def project_to_nonnegative(ff):
    tmp = ff.vector()[:]
    tmp[tmp<0] = 0
    ff.vector()[:] = tmp
    
    