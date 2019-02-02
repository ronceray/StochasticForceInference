import matplotlib.pyplot as plt
import numpy as np





def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3] 
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def ims(I,val=None,**kwargs):
    """ An imshow helper class with better default params. """
    if val is None:
        val = abs(I).max()
    plt.imshow(I,interpolation='none',cmap='RdBu',vmin=-val,vmax=val,**kwargs)
def imcmp(I,J):
    """ Quick visual comparison between two arrays. """
    val = max(abs(I).max(),abs(J).max())
    plt.subplot(121)
    ims(I,val)
    plt.subplot(122)
    ims(J,val)
    plt.show()



def comparison_scatter(Xexact,Xinferred,vmax=None,color='cornflowerblue',alpha=0.05,axes=None,y=0.8):
    """ This method is used to compare the inferred force components to
    the exact ones along the trajectory, in a graphical way. It assumes
    that the compute_accuracy method has been called before to provide 
    the exact force components."""

    # Flatten the data:
    Xe = np.array([ x.reshape(int(np.prod(x.shape))) for x in Xexact ])
    Xi = np.array([ x.reshape(int(np.prod(x.shape))) for x in Xinferred ])
    Xe = Xe.reshape(np.prod(Xe.shape))
    Xi = Xi.reshape(np.prod(Xi.shape))
    
    if vmax is None:
        vmax = max(abs(Xe).max(),abs(Xi).max())
    range_vals=np.array([[-vmax,vmax],[-vmax,vmax]]) 
    plt.scatter(Xe,Xi,alpha=alpha,linewidth=0,c=color)

    from scipy.stats import pearsonr
    (r,p) =  pearsonr(Xe,Xi)
    plt.plot([-vmax,vmax],[-vmax,vmax],'k-')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('exact')
    plt.ylabel('inferred')
    plt.title(r"$r="+str(round(r,2 if r<0.98 else 3 if r<0.999 else 4  if r<0.9999  else 5))+"$",loc='left',y=y,x=0.05,fontsize=10)
    plt.xticks([0.])
    plt.yticks([0.])
    plt.xlim(-vmax,vmax)
    plt.ylim(-vmax,vmax) 

 

# These functions are designed to help build simple processes.

def random_definite_positive_matrix(dim):
    U = np.random.normal(size=(dim,dim))
    S = U.dot(U.T)
    return S * dim / np.linalg.norm(S)

def random_matrix(dim):
    U = np.random.normal(size=(dim,dim))
    return U * dim / np.linalg.norm(U)

def random_symmetric_matrix(dim):
    U = np.random.normal(size=(dim,dim))
    S = (U + U.T)
    return S * dim / np.linalg.norm(S)

def random_antisymmetric_matrix(dim):
    V = np.random.normal(size=(dim,dim))
    return dim * (V - V.T) / np.linalg.norm(V-V.T)

