import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.linalg import LinAlgError


class StochasticTrajectoryData(object):
    """This class is a formatter and wrapper class for stochastic
    trajectories. It performs basic operations (discrete derivative,
    mid-point "Stratonovich" positions), and provides plotting
    functions for the trajectory and inferred force and diffusion
    fields (2D with matplotlib, 3D with mayavi2).

    """ 
    def __init__(self,data,t,data_type='homogeneous'):
        """If data_type is 'homogeneous', data is a list of length Nsteps of
        Nparticles x d numpy arrays, where d is the dimensionality of
        phase space, Nsteps the length of the trajectory, and
        Nparticles the number of particles (assumed identical).

        If it is 'heterogeneous', this corresponds to trajectories
        where the number of particles varies with time, for
        instance. In this case we'll assume that the analysis has been
        done before, and that data is a dictionary with keys dX,
        X_ito, X_strat and dt. For each time point the arrays dX,
        X_ito and X_strat must have the same dimensions.

        """
        if data_type == 'homogeneous':
            Nparticles,self.d = data[0].shape
            self.t = t[1:-1]
            self.dt = t[2:] - t[1:-1] 

            self.X_ito = 1. * data[1:-1] 
            self.dX = data[2:] - data[1:-1]
            self.dX_pre = data[1:-1] - data[:-2]
            self.X_strat = self.X_ito + 0.5 * self.dX
            self.Xdot = np.einsum('tmi,t->tmi',self.dX, 1. /self.dt)
            self.Nparticles = [ X.shape[0] for X in self.X_strat ] 
            # Total time-per-particle in the trajectory
            self.tauN = np.einsum('t,t->',self.dt,self.Nparticles)


        elif data_type == 'heterogeneous':
            self.t = t
            self.X_ito = data['X_ito']
            self.dX = data['dX']
            self.dX_pre = data['dX_pre']
            self.dt = data['dt']
            self.X_strat = [ self.X_ito[t] + 0.5 * self.dX[t] for t,dt in enumerate(self.dt) ]
            self.d = self.X_ito[0].shape[1]
            self.Nparticles = [ X.shape[0] for X in self.X_strat ] 
            self.tauN = np.einsum('t,t->',self.dt,self.Nparticles)
            self.Xdot = [  self.dX[t] / dt for t,dt in enumerate(self.dt) ]
            
        else:
            raise KeyError("Wrong data_type")

    def inner_product_empirical(self,f,g,integration_style='Stratonovich'):
        # Time integration of two functions (with indices) of the
        # position along the trajectory, with averaging over particle
        # indices.

        # Two ways of calling it (can be mixed):
        #  - with an array (t-1) x Nparticles x dim (eg for Xdot)
        
        #  - with a dxN-dimensional callable function f(X) (eg for the
        #    moments); will apply it on all time-points and particles,
        #    with corresponding arguments.
        if integration_style=='Stratonovich':
            X = self.X_strat
            dt = self.dt
        elif integration_style=='Ito':
            X = self.X_ito
            dt = self.dt
        elif integration_style=='StratonovichTruncated':
            X = self.X_strat[1:-1]
            dt = self.dt[1:-1]
        else:
            raise KeyError("Wrong integration_style keyword.")
        func = lambda t : np.einsum('im,in->mn', f(X[t]) if callable(f) else f[t] , g(X[t]) if callable(g) else g[t] )
        return self.trajectory_integral(func,dt)

    def trajectory_integral(self,func,dt=None):
        # A helper function to perform trajectory integrals with
        # constant memory use. The function takes integers as
        # argument.
        if dt is None:
            dt = self.dt
        result = 0. # Generic initialization - works with any
                    # array-like function.
        for t,ddt in enumerate(dt):
            result += ddt * func(t)
        return result / self.tauN

    
    def plot_process(self,dir1=None,dir2=None,shift=(0,0),tmin=None,tmax=None, cmap = 'viridis',particle=0,**kwargs):
        """Basic 2D plotting of the trajectory. The color gradient indicates
        time. dir1 and dir2 arguments, if specified, should be two
        orthogonal unit vectors, defining the projection plane of the
        representation. 'particle' argument indicates which particle
        should be considered, if relevant.
        """
        if dir1 is None:
            dir1 = axisvector(0,self.d)
        if dir2 is None:
            dir2 = axisvector(1,self.d)
        x = np.array([ dir1.dot(u[particle,:]) for u in self.X_ito[tmin:tmax] ])
        y = np.array([ dir2.dot(u[particle,:]) for u in self.X_ito[tmin:tmax] ])
        plt.quiver(x[:-1]+shift[0],y[:-1]+shift[1],x[1:]-x[:-1],y[1:]-y[:-1],self.t[tmin:tmax][:-1] ,cmap=cmap,
                   headaxislength = 0.0,headwidth = 0., headlength = 0.0, minlength=0., minshaft = 0.,  scale = 1.0,units = 'xy',lw=0.,**kwargs)
        plt.axis('equal') 
        plt.xticks([])
        plt.yticks([])
    

    def plot_field(self,dir1=None,dir2=None,field=None,center=None,N = 10,scale=1.,autoscale=False,color='g',radius=None,positions = None,**kwargs):
        """Plot a 2D vector field (or a 2D slice of a higher-dimensional
        field), either as a mesh or on a specified list of points. The
        'field' parameter is a lambda function X -> F(X), for instance.
        """
        if dir1 is None:
            dir1 = axisvector(0,self.d)
        if dir2 is None:
            dir2 = axisvector(1,self.d)
        if center is None:
            center = self.X_ito.mean(axis=(0,1))
        if radius is None:
            radius  = 0.5 * ( self.X_ito.max(axis=(0,1)) - self.X_ito.min(axis=(0,1)) ).max()

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
            v = field( pos.reshape((1,self.d)) )
            vX.append(dir1.dot(v[0,:]))
            vY.append(dir2.dot(v[0,:]))

        if autoscale:
            scale /= max(np.array(vX)**2 + np.array(vY)**2)**0.5
            print("Vector field scale:",scale)
        plt.quiver(gridX,gridY,scale*np.array(vX),scale*np.array(vY) ,scale = 1.0,units = 'xy',color = color,minlength=0.,**kwargs)
        plt.ylim(-radius+dir2.dot(center),radius+dir2.dot(center))
        plt.xlim(-radius+dir1.dot(center),radius+dir1.dot(center))
        plt.axis('equal') 
        plt.xticks([])
        plt.yticks([])


    def plot_tensor_field(self,field,center=None,N = 10,scale=1.,autoscale=False,color='g',radius=None,positions=None,**kwargs):
        """ Plot a tensor field for 2D processes. """
        if center is None:
            center = self.X_ito.mean(axis=(0,1))
        if radius is None:
            radius  = 0.5 * ( self.X_ito.max(axis=(0,1)) - self.X_ito.min(axis=(0,1)) ).max()

        if positions is None:
            positions = []
            for a in np.linspace(-radius,radius,N):
                for b in np.linspace(-radius,radius,N):
                    positions.append(center + np.array([a,b]))

        X,Y,U,V = [],[],[],[]
        for pos in positions: 
            posr = pos.reshape((1,self.d)) 
            tensor = field( posr )
            eigvals,eigvecs = np.linalg.eigh(tensor.reshape((2,2)))
            for j in range(2): 
                X.append( pos[0] )
                Y.append( pos[1] )
                U.append( eigvals[j] * eigvecs[0,j] )
                V.append( eigvals[j] * eigvecs[1,j] )

        if autoscale:
            scale /= max(np.array(U)**2 + np.array(V)**2)**0.5
            print("Tensor field scale:",scale)

        X,Y =  np.array(X),np.array(Y)
        dX,dY = 0.5*scale*np.array(U),0.5*scale*np.array(V)
                                            
        plt.quiver(X-dX,Y-dY,2*dX,2*dY,scale = 1.0,units = 'xy',color = color,minlength=0.,headwidth=1.,headlength=0.,**kwargs)
                                                 
        plt.axis('equal') 
        plt.xticks([])
        plt.yticks([])


    def plot_process_3D(self,dir1=None,dir2=None,dir3=None,clear=False,particle=0):
        """3D visualization of the trajectory using Mayavi; shading indicates
        time."""
        if dir1 is None:
            dir1 = axisvector(0,self.d)
        if dir2 is None:
            dir2 = axisvector(1,self.d)
        if dir3 is None:
            dir3 = axisvector(2,self.d)
        import mayavi.mlab
        if clear:
            mayavi.mlab.clf()
        f = mayavi.mlab.gcf()
        f.scene.background = (1.,1.,1.)
        x = np.array([ dir1.dot(u[particle,:]) for u in self.X_ito ])
        y = np.array([ dir2.dot(u[particle,:]) for u in self.X_ito ])
        z = np.array([ dir3.dot(u[particle,:]) for u in self.X_ito ])
        mayavi.mlab.plot3d(x,y,z,self.t,colormap='bone')


    def plot_field_3D(self,field,dir1=None,dir2=None,dir3=None,center=None,Npts = 10,scale=1.,autoscale=False,color='g',radius=None,positions = None,cmap='summer',cval=0.,**kwargs):
        # Display a 3D vector field (requires Mayavi2 package)
        import mayavi.mlab
        f = mayavi.mlab.gcf()
        f.scene.background = (1.,1.,1.)

        if dir1 is None:
            dir1 = axisvector(0,self.d)
        if dir2 is None:
            dir2 = axisvector(1,self.d)
        if dir3 is None:
            dir3 = axisvector(2,self.d)
        if center is None:
            center = self.X_ito.mean(axis=(0,1))
        if radius is None:
            radius  = 0.5 * ( self.X_ito.max(axis=(0,1)) - self.X_ito.min(axis=(0,1)) ).max()

        if positions is None:
            positions = []
            for a in np.linspace(-radius,radius,Npts):
                for b in np.linspace(-radius,radius,Npts):
                    for c in np.linspace(-radius,radius,Npts):
                        positions.append(center + a * dir1 + b * dir2 + c * dir3 )

        gridX,gridY,gridZ = [],[],[]
        vX,vY,vZ = [],[],[]
        for pos in positions:
            x = dir1.dot(pos)
            y = dir2.dot(pos)
            z = dir3.dot(pos)
            gridX.append(x)
            gridY.append(y)
            gridZ.append(z)
            # Reshape to adapt to the SFI data structure (1st index is
            # particle index):
            v = field( pos.reshape((1,self.d)) )
            vX.append(dir1.dot(v[0,:]))
            vY.append(dir2.dot(v[0,:]))
            vZ.append(dir3.dot(v[0,:]))

        if autoscale:
            scale /= max(np.array(vX)**2 + np.array(vY)**2 + np.array(vZ)**2)**0.5
            print("Vector field scale:",scale)
        
        mayavi.mlab.quiver3d(gridX,gridY,gridZ,scale*np.array(vX),scale*np.array(vY),scale*np.array(vZ),scalars=0.*np.array(vZ)+cval,vmin=0.,vmax=1.,colormap=cmap,mode='arrow', scale_factor = 2.,scale_mode = 'vector', resolution = 8 )


    def plot_tensor_field_3D(self,field=None,center=None,Npts = 7,scale=1.,autoscale=True,color='g',radius=None,positions = None,cmap='summer',cval=0.,**kwargs):
        # Display a 3D tensor field (requires Mayavi2 package)
        import mayavi.mlab
        f = mayavi.mlab.gcf()
        f.scene.background = (1.,1.,1.)
            
        if center is None:
            center = self.X_ito.mean(axis=(0,1))
        if radius is None:
            radius  = 0.5 * ( self.X_ito.max(axis=(0,1)) - self.X_ito.min(axis=(0,1)) ).max()

        if positions is None:
            positions = []
            for a in np.linspace(-radius,radius,Npts):
                for b in np.linspace(-radius,radius,Npts):
                    for c in np.linspace(-radius,radius,Npts):
                        positions.append(center + np.array([a,b,c]))


        X,Y,Z,U,V,W = [],[],[],[],[],[]
        for pos in positions: 
            posr = pos.reshape((1,self.d)) 
            tensor = field( posr )
            eigvals,eigvecs = np.linalg.eigh(tensor.reshape((3,3)))
            for j in range(3): 
                X.append( pos[0] )
                Y.append( pos[1] )
                Z.append( pos[2] )
                U.append( eigvals[j] * eigvecs[0,j] )
                V.append( eigvals[j] * eigvecs[1,j] )
                W.append( eigvals[j] * eigvecs[2,j] )

        if autoscale:
            scale /= max(np.array(U)**2 + np.array(V)**2 + np.array(W)**2)**0.5
            print("Tensor field scale:",scale)

       
        X,Y,Z =  np.array(X),np.array(Y),np.array(Z)
        dX,dY,dZ = 0.5*scale*np.array(U),0.5*scale*np.array(V),0.5*scale*np.array(W)

        mayavi.mlab.quiver3d(X-dX,Y-dY,Z-dZ, 2*dX, 2*dY, 2*dZ,scalars=0.*np.array(Z)+cval,vmin=0.,vmax=1.,colormap=cmap,mode='2ddash', scale_factor = 1.,scale_mode = 'vector', resolution = 8 ,**kwargs)

        
    def plot_particles(self,a=0,b=1,X=None,t=-1,colored=True,active=False,u=0.35,**kwargs):
        # Display the position of all particles at time t.
        if X is None:
            X = self.X_ito[t]
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

    def plot_particles_field(self,field,X=None,t=-1,dir1=None,dir2=None,center=None,radius=None,scale=1.,autoscale=False,color='g',**kwargs):

        if dir1 is None:
            dir1 = axisvector(0,self.d)
        if dir2 is None:
            dir2 = axisvector(1,self.d)

        # Plot a 2D vector field
        if X is None:
            X = self.X_ito[t]
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
 

