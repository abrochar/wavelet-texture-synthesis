import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.transform
import scipy.io as sio

def get_color_mat(name,ks,size):
    assert(ks==0) # start id
    data = sio.loadmat('./data/' + name + '.mat')
    imori = data['im']
    return imori

def get_gray_mat(name,ks,size):
    data = sio.loadmat('./data/' + name + '.mat')
    imori = data['img']
    assert(ks==0)
    imori = imori[np.newaxis]
    return imori

class prep_image01():
    def __init__(self):
        self.mean = None

    def forward(self,im):
        # im: (...,size,size)
        # texture: (...,size,size)
        if self.mean is None:
            m1 = np.mean(im,axis=1,keepdims=True)
            m2 = np.mean(m1,axis=2,keepdims=True)
            self.mean = m2
            #print(self.mean)
        return im - self.mean

    def backward(self,im):
        return im + self.mean        

# show radial spectrum
def compute_radial_sp(sp2,N):
    N2 = N//2
    rk = np.zeros(N2+1)
    om = np.linspace(-N2,N2-1,N)
    om1, om2 = np.meshgrid(om, om)
    mod_om = np.sqrt(om1**2 + om2**2)
    mod_om = np.fft.fftshift(mod_om)
    if 0:
        plt.imshow(sp2)
        plt.show()        
    if 0:
        plt.imshow(mod_om)
        plt.show()
    for k in range(0,N2+1):
        mask = (mod_om>=k) & (mod_om<k+1)
        if 0:
            print(k,np.sum(mask))
        rk[k] = np.mean(sp2[mask])
    return rk, np.arange(0,N2+1)


def show_np_color_image(imori, cmin=0, cmax=1):
    # assume imori is from get_color_image
    im = imori.transpose((1,2,0))
    #im = im+0.5
    plt.imshow(im,cmap='jet', vmin=cmin, vmax=cmax)
    plt.show()


def range_np_color_image(imori):
    im = imori # .transpose((1,2,0))
    #im = im+0.5
    return np.quantile(im,0.01), np.quantile(im,0.99)

def compare_np_color_images_ingray(imori,imrec,ori=True):
    if ori:
        plt.imshow(imori[0,:,:],vmin=0,vmax=1,cmap='gray')
        plt.colorbar()
        plt.show()
    plt.imshow(imrec[0,:,:],vmin=0,vmax=1,cmap='gray')
    plt.colorbar()
    plt.show()
    
    if ori:
        plt.imshow(imori[1,:,:],vmin=0,vmax=1,cmap='gray')
        plt.colorbar()    
        plt.show()
    plt.imshow(imrec[1,:,:],vmin=0,vmax=1,cmap='gray')
    plt.colorbar()        
    plt.show()
    
    if ori:
        plt.imshow(imori[2,:,:],vmin=0,vmax=1,cmap='gray')
        plt.colorbar()    
        plt.show()   
    plt.imshow(imrec[2,:,:],vmin=0,vmax=1,cmap='gray')
    plt.colorbar()
    plt.show()


def plot2pdf(img,pdfname,cmin=-0.5,cmax=0.5,cmap0='gray',asp='equal'):
    fig = plt.figure() # frameon=False)
    sizes = img.shape # [0],img.shape[1])
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img,cmap=cmap0,aspect=asp,vmin=cmin,vmax=cmax)
    if asp is not 'equal':
        forceAspect(ax,aspect=asp) 
    print('save to pdf file', pdfname)
    plt.savefig(pdfname+'.pdf',dpi=sizes[0],cmap=cmap0) # , bbox_inches='tight')
    plt.show()

def plot2pdfcbar(img,pdfname,cmin=-0.5,cmax=0.5,cmap0='gray',axislabel=None,cbar=True):
    fig = plt.figure()
    #sizes = img.shape # [0],img.shape[1])
    #fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    ima = plt.imshow(img,vmin=cmin,vmax=cmax,aspect='equal',interpolation='none')
    ima.set_cmap(cmap0)
    if cbar:
        plt.colorbar()
    if axislabel is None:
        plt.axis('off')
    else:
        plt.xlabel(axislabel['x'],fontsize=20)
        h = plt.ylabel(axislabel['y'],fontsize=20)
        h.set_rotation(0)
    fig.tight_layout()
    print('save to pdf file', pdfname)
    plt.savefig(pdfname+'.pdf',bbox_inches='tight', pad_inches = 0) # cmap=cmap0)
    plt.show()
