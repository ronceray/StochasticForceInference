import matplotlib.pyplot as plt
import numpy as np

""" Utility classes for plotting the results of SFI, used only to
display the results of the example files.

"""

def comparison_scatter(Xexact,Xinferred,error=None,maxpoints=10000,vmax=None,color=None,alpha=0.05,y=0.8,mode='both',fontsize=9):
    """This method is used to compare inferred components to the
    exact ones along the trajectory, in a graphical way.

    Xexact, Xinferred: jnp arrays (Nsteps,...); must have the same shape.

    error: predicted standard deviation for X_inferred.

    maxpoints: if Nsteps / 2 * maxpoints, data will be subsampled.
    
    """

    subsample = max(1,Xexact.shape[0]//maxpoints)
    # Flatten the data:
    Xe = np.array(Xexact)[::subsample].reshape(-1)
    Xi = np.array(Xinferred)[::subsample].reshape(-1)

    MSE = sum((Xe-Xi)**2) / sum(Xe**2 + Xi**2)
    
    if vmax is None:
        vmax = max(abs(Xe).max(),abs(Xi).max())
    plt.scatter(Xe,Xi,alpha=alpha,linewidth=0,c=color)

    if error is not None:
        xvals = np.array([-vmax,vmax])
        confidence_interval = 2*error**0.5 * Xi.std()
        plt.plot(xvals,xvals+confidence_interval,"k:")
        plt.plot(xvals,xvals-confidence_interval,"k:")
    from scipy.stats import pearsonr
    (r,p) =  pearsonr(Xe,Xi)
    plt.plot([-1e10,1e10],[-1e10,1e10],'k-')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.axis('equal')
    plt.xlabel('exact')
    plt.ylabel('inferred')

    titlestring = ""
    if mode == 'r' or mode == 'both' :
        titlestring += r"$r="+str(round(r,2 if r<0.98 else 3 if r<0.999 else 4  if r<0.9999  else 5))+"$"
    if mode == "both":
        titlestring += "\n"
    if mode == 'MSE' or mode == 'both' :
        titlestring += "MSE="+str(round(MSE,3))
    plt.title(titlestring,loc='left',y=y,x=0.05,fontsize=fontsize)
    plt.xticks([0.])
    plt.yticks([0.])
    plt.xlim(-vmax,vmax)
    plt.ylim(-vmax,vmax) 

import matplotlib.collections as mcoll

def plot_process(data, dir1=None, dir2=None, shift=(0,0), tmin=0, tmax=-1, 
                 particles=None, dx_minus_too=False, cmap='viridis', linewidth=1.5, alpha=1.0, plot_colorbar=False):
    """
    Plot a 2D projection of a trajectory with color-coded segments for multiple particles.

    Parameters
    ----------
    data : StochasticTrajectoryData
        The trajectory data container, with data.X of shape (T, Nparticles, d).
    dir1 : array-like, shape (d,), optional
        A unit vector for the horizontal axis of the plot.
    dir2 : array-like, shape (d,), optional
        A unit vector for the vertical axis of the plot.
    shift : tuple of float
        (xshift, yshift) to shift the entire plot.
    tmin, tmax : int, optional
        Index range for time steps to plot. Defaults to entire range.
    particles : list or None, optional
        List of particle indices to plot. If None, plots all particles.
    dx_minus_too : bool, optional
        Whether to also plot the increments from the prior step (for discontinuous trajectories).
    cmap : str, optional
        The colormap to use for the time-based coloring.
    linewidth : float, optional
        Width of the plotted trajectory lines.
    alpha : float, optional
        Transparency of the plotted lines.
    """

    if dir1 is None:
        dir1 = axisvector(0, data.d)
    if dir2 is None:
        dir2 = axisvector(1, data.d)

    if tmin is None:
        tmin = 0
    if tmax is None:
        tmax = data.X.shape[0]

    # Get total number of particles
    Nparticles = data.X.shape[1]
    if particles is None:
        particles = np.arange(Nparticles)  # Default: plot all particles
    else:
        particles = np.array(particles)

    # Extract trajectory positions and displacements for all selected particles
    X = np.array(data.X[tmin:tmax, particles, :])  # Shape: (T, Nparticles, d)
    dX = np.array(data.dX[tmin:tmax, particles, :])  # Shape: (T, Nparticles, d)

    # Compute projected positions
    x, y = X @ dir1, X @ dir2  # Shape: (T, Nparticles)
    dx, dy = dX @ dir1, dX @ dir2  # Shape: (T, Nparticles)

    # Reshape into a single list of segments (each segment is [start, end])
    points_start = np.column_stack([x.ravel(order='F') + shift[0], y.ravel(order='F') + shift[1]])
    points_end = np.column_stack([x.ravel(order='F') + dx.ravel(order='F') + shift[0], y.ravel(order='F') + dy.ravel(order='F') + shift[1]])
    segments = np.stack([points_start, points_end], axis=1)

    # Normalize color by time
    norm = plt.Normalize(data.t[tmin], data.t[tmax])

    # Create LineCollection in one go
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(np.tile(data.t[tmin:tmax], len(particles)))  # Set time as color

    # Set up the figure
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    if plot_colorbar:
        plt.colorbar(lc, label='Time')

    # Handle dx_minus_too for previous steps (optional)
    if dx_minus_too and hasattr(data, 'dX_minus'):
        dX_minus = np.array(data.dX_minus[tmin:tmax, particles, :])
        dxm, dym = dX_minus @ dir1, dX_minus @ dir2  # Shape: (T, Nparticles)

        # Reshape previous increments
        points_m_start = np.column_stack([x.ravel() - dxm.ravel() + shift[0], y.ravel() - dym.ravel() + shift[1]])
        points_m_end = np.column_stack([x.ravel() + shift[0], y.ravel() + shift[1]])
        segments_m = np.stack([points_m_start, points_m_end], axis=1)

        # LineCollection for previous increments
        lc_m = mcoll.LineCollection(segments_m, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha * 0.7)
        lc_m.set_array(np.tile(data.t[tmin:tmax], len(particles)))  # Color by time
        ax.add_collection(lc_m)

    plt.show()
    
def plot_field(data,dir1=None,dir2=None,field=None,center=None,N = 10,scale=1.,autoscale=False,color='g',radius=None,powernorm = 0,positions = None,**kwargs):
    """Plot a 2D vector field (or a 2D slice of a higher-dimensional
    field), either as a mesh or on a specified list of points. The
    'field' parameter is a lambda function X -> F(X), for instance.
    """
    if dir1 is None:
        dir1 = axisvector(0,data.d)
    if dir2 is None:
        dir2 = axisvector(1,data.d)
    if center is None:
        center = data.X.mean(axis=(0,1))
    if radius is None:
        radius  = 0.5 * ( data.X.max(axis=(0,1)) - data.X.min(axis=(0,1)) ).max()

    if positions is None:
        positions = []
        for a in np.linspace(-radius,radius,N):
            for b in np.linspace(-radius,radius,N):
                positions.append(center + a * dir1 + b * dir2 )

    gridX,gridY = [],[]
    vX,vY = [],[]
    for pos in positions:
        x = dir1.dot(pos)
        y = dir2.dot(pos)
        gridX.append(x)
        gridY.append(y)
        # Reshape to adapt to the SFI data structure (1st index is
        # particle index):
        v = field( pos.reshape((1,data.d)) )
        v /= np.linalg.norm(v)**powernorm
        vX.append(dir1.dot(v[0,:]))
        vY.append(dir2.dot(v[0,:]))

    if autoscale:
        scale /= max(np.array(vX)**2 + np.array(vY)**2)**0.5
    plt.quiver(gridX,gridY,scale*np.array(vX),scale*np.array(vY) ,scale = 1.0,units = 'xy',color = color,minlength=0.,**kwargs)
    plt.ylim(-radius+dir2.dot(center),radius+dir2.dot(center))
    plt.xlim(-radius+dir1.dot(center),radius+dir1.dot(center))
    plt.axis('equal') 
    plt.xticks([])
    plt.yticks([])


def plot_tensor_field(data,field,center=None,N = 10,scale=1.,autoscale=False,color='g',radius=None,positions=None,**kwargs):
    """ Plot a tensor field for 2D processes. """
    if center is None:
        center = data.X.mean(axis=(0,1))
    if radius is None:
        radius  = 0.5 * ( data.X.max(axis=(0,1)) - data.X.min(axis=(0,1)) ).max()

    if positions is None:
        positions = []
        for a in np.linspace(-radius,radius,N):
            for b in np.linspace(-radius,radius,N):
                positions.append(center + np.array([a,b]))

    X,Y,U,V = [],[],[],[]
    for pos in positions: 
        posr = pos.reshape((1,data.d)) 
        tensor = field( posr )
        eigvals,eigvecs = np.linalg.eigh(tensor.reshape((2,2)))
        for j in range(2): 
            X.append( pos[0] )
            Y.append( pos[1] )
            U.append( eigvals[j] * eigvecs[0,j] )
            V.append( eigvals[j] * eigvecs[1,j] )

    if autoscale:
        scale /= max(np.array(U)**2 + np.array(V)**2)**0.5

    X,Y =  np.array(X),np.array(Y)
    dX,dY = 0.5*scale*np.array(U),0.5*scale*np.array(V)

    plt.quiver(X-dX,Y-dY,2*dX,2*dY,scale = 1.0,units = 'xy',color = color,minlength=0.,headwidth=1.,headlength=0.,**kwargs)

    plt.axis('equal') 
    plt.xticks([])
    plt.yticks([])




def plot_particles(data,a=0,b=1,t=-1,colored=True,active=False,u=0.35,**kwargs):
    # Display the position of all particles at time t.
    X = data.X[t]
    x = X[:,a]
    y = X[:,b]
    if colored:
        plt.scatter(x,y,cmap='magma',s=100,c=np.linspace(0,1,len(X)),**kwargs)
    else:
        plt.scatter(x,y,s=100,c='w',edgecolor='k',**kwargs)
    if active:
        # Show the orientations of active Brownian particles.
        xa = x + u*np.cos(X[:,2])
        ya = y + u*np.sin(X[:,2])
        plt.scatter(xa,ya,c='k',s=20,**kwargs)

    plt.axis('equal')  
    plt.xticks([])
    plt.yticks([])

def plot_particles_field(data,field,t=-1,dir1=None,dir2=None,center=None,radius=None,scale=1.,autoscale=False,color='g',**kwargs):

    if dir1 is None:
        dir1 = axisvector(0,data.d)
    if dir2 is None:
        dir2 = axisvector(1,data.d)

    # Plot a 2D vector field
    X = data.X[t]
    F = field(X)
    if center is None: 
        center = X.mean(axis=0)
    if radius is None:
        radius  = 0.5 * ( X.max(axis=0) - X.min(axis=0) ).max()

    gridX,gridY = [],[]
    vX,vY = [],[]
    for ind,pos in enumerate(X):
        x = dir1.dot(pos)
        y = dir2.dot(pos)
        gridX.append(x)
        gridY.append(y)
        vX.append(dir1.dot(F[ind,:]))
        vY.append(dir2.dot(F[ind,:]))

    if autoscale:
        scale /= max(np.array(vX)**2 + np.array(vY)**2)**0.5
        print("Vector field scale:",scale)
    plt.quiver(gridX,gridY,scale*np.array(vX),scale*np.array(vY) ,scale = 1.0,units = 'xy',color = color,minlength=0.,**kwargs)
    plt.ylim(-radius+dir2.dot(center),radius+dir2.dot(center))
    plt.xlim(-radius+dir1.dot(center),radius+dir1.dot(center))
    plt.axis('equal') 
    plt.xticks([])
    plt.yticks([])

def axisvector(index,dim):
    """d-dimensional vector pointing in direction index."""
    return np.array([ 1. if i == index else 0. for i in range(dim)])
 

