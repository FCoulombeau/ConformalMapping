# -*- coding: utf-8 -*-
"""Conformal mappings of the complex plane used to manipulate and transform
images and videos.
@author: FranÃ§ois Coulombeau
@date : 2014
@version : 0.2
"""

import matplotlib.pyplot as plt
import moviepy.editor as vided
import numpy as np
import math as m
import time

normcar=lambda u:u.real**2+u.imag**2
arg=lambda u:np.arctan2(u.imag,u.real)
ln=lambda u:np.log(normcar(u))/2.+1j*arg(u)
scale_rotate=lambda r,t:r*(m.cos(t*2*m.pi)+1j*m.sin(t*2*m.pi))

def sign(x):
    """Returns 1 if the argument is positive, -1 else"""
    if x>=0:return 1
    return -1

def mirror(d,X=2,Y=2,nbpix=0,color=[255,255,255]):
    """Returns the numpy array representing the output image got by mirroring
    the input d along its right edge if X=2 and along its bottom edge if Y=2.
    Puts stripes with the number "nbpix" of pixels around the output and
    between the mirrored inputs with color "color".
    """
    li,co=d.shape[0:2]
    r=np.ndarray((li*Y+(Y+1)*nbpix,X*co+(X+1)*nbpix,d.shape[2]),dtype=d.dtype)
    r[:nbpix,:]=color
    r[:,:nbpix]=color
    r[li+nbpix:li+2*nbpix,:]=color
    r[:,co+nbpix:co+2*nbpix]=color
    r[nbpix:li+nbpix,nbpix:co+nbpix]=d[:,:]
    if Y==2:
        if nbpix>0:
            r[li+2*nbpix:-nbpix,:]=r[li+nbpix-1:(2-Y)*(li+nbpix)+nbpix-1:-1,:]
            r[-nbpix:,:]=color
        else:
            r[li:,:]=r[li-1::-1,:]
    if X==2:
        if nbpix>0:
            r[:,co+2*nbpix:-nbpix]=r[:,co+nbpix-1:(2-X)*(co+nbpix)+nbpix-1:-1]
            r[:,-nbpix:]=color
        else:
            r[:,co:]=r[:,co-1::-1]
    return r
    
def morphing(im1,wt1,im2,wt2):
    """Returns a mix of the images im1 with weight wt1 and im2 with weight wt2,
    each image given as a numpy array.
    """
    return ((im1[:,:,:])*wt1+(im2[:,:,:])*wt2)/(wt1+wt2)

def halfhalf(im1,im2):
    """Returns the image made of half im1 and half im2, horizontally.
    im1 and im2 must have the same height, and im2 must be wider than im1.
    """
    im=np.ndarray(im1.shape)
    im[:,:im1.shape[1]//2,:]=im1[:,:im1.shape[1]//2,:]
    im[:,im1.shape[1]//2:,:]=im2[:,im1.shape[1]//2:,:]
    return im

def _change_origin(foncInv,c,d):
    return lambda u:foncInv(u)/c+d

def somme(l):
    """Takes an array l and returns the array made of the sums of all subarrays
    over the first dimension along."""
    res=l[0]
    for k in l[1:]: res+=k
    return res

def _power(Exposant,c=1.,d=0.):
    return (lambda u:np.power(normcar(u),1./2/Exposant)*(np.cos(arg(u)/Exposant)+1j*np.sin(arg(u)/Exposant)),c,d)

def _second_degree_polynomial(a,b,c=1.,d=0.):
    return (lambda u:normcar(a**2+4*(b-u))**(1./4)*(np.cos(arg(a**2+4*(b-u))/2)+1j*np.sin(arg(a**2+4*(b-u))/2)),c,d)

def _inverse_polynomial(l,c=1.,d=0.):
    return (lambda u:somme([l[k]*np.power(u,k) for k in range(len(l))]),c,d)

def _complex_exponential(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):    
    if Angle:
        alpha=m.atan2((-1)**Q*P,N*forme)    
        c2=(-1)**(Q//2)*m.pi/P*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    else:
        c2=c
    return (ln,c2,d)

def _complex_sine(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*m.pi/P/2*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    else:
        c2=c
    return (lambda u:ln(u+_power(2,c2,d)[0](1+u**2)),c2,d)

def _squareroot_sine(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*m.pi/P/2*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    else:
        c2=c
    return (lambda u:ln(u**2+_power(2,c2,d)[0](1+u**4)),c2,d)

def _complex_tangent(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=m.atan2((-1)**Q*P,N*forme)    
        c2=(-1)**(Q//2)*m.pi/P/2*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    else:
        c2=c
    return (lambda u:(-ln(1-u)+ln(1+u))/2,c2,d)

def _symmetry4(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=m.atan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*m.pi/P/2*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    else:
        c2=c
    return (lambda u:(-ln(1-u)+ln(1+u)-ln(1j-u)+ln(1j+u))/2,c2,d)
    
def _symmetry4v2(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=m.atan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*m.pi/P/2*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    else:
        c2=c
    return (lambda u:(ln(-1+u)+ln(1+u)+ln(1j+u)+ln(-1j+u))/2,c2,d)
    
def _symmetry3(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=m.atan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*m.pi/P/2*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    else:
        c2=c
    return (lambda u:(ln(1+u)+ln(-1/2+m.sqrt(3)/2*1.j+u)+ln(-1/2-m.sqrt(3)/2*1.j+u))/2,c2,d)
    
def _complex_arcsine(c=1.,d=0.):
    return (lambda u:np.sinh(u.real)*np.cos(u.imag)+1j*np.sin(u.imag)*np.cosh(u.real),c,d)

def _complex_arctan(c=1.,d=0.):
    return (lambda u:(np.tanh(u.real)+1j*np.tan(u.imag))/(1+np.tanh(u.real)*1j*np.tan(u.imag)),c,d)
    
def _complex_logarithm(c=1.,d=0.):
    return (lambda u:np.exp(u.real)*(np.cos(u.imag)+1j*np.sin(u.imag)),c,d)
    
def _oval(forme,N=1,P=1,Q=0,d=0.):
    alpha=m.atan2((-1)**Q*P,N*forme)    
    c2=(-1)**(Q//2)*m.pi/P/2*m.sin(alpha)*(1.j*m.cos(alpha)+m.sin(alpha))
    cc2=m.sinh(m.pi*m.cos(m.pi/2-2*alpha)/2)*m.cos(m.pi*m.sin(m.pi/2-2*alpha)/2)+1j*m.cosh(m.pi*m.cos(m.pi/2-2*alpha)/2)*m.sin(m.pi*m.sin(m.pi/2-2*alpha)/2)
    g2=lambda u:ln(u+_power(2,c2,d)[0](1+u**2))
    return (lambda u:g2(g2(u)*cc2/1j*2/m.pi),c2,d)

def _semi_scaling(r):
    return lambda u:u.real/r+1j*u.imag

def _fisheye(aff=1.,c=1.,d=0.):
    return (lambda u:_semi_scaling(1/aff)(_power(2)[0](_complex_sine(1.,Angle=False)[0](_power(1/2)[0](_semi_scaling(aff)(u))*m.pi/4)/m.pi*4)),c,d)

def _invers_fisheye(aff=1.,c=1.,d=0.):
    return (lambda u:_semi_scaling(1/aff)(_power(2)[0](_complex_arcsine()[0](_power(1/2)[0](_semi_scaling(aff)(u))*m.pi/4)/m.pi*4)),c,d)

def _equirectangular(c=1.,d=0.):
    return (lambda u:np.tan(m.pi*u.imag/4+m.pi/4)*(1j*np.cos(m.pi*u.real/2)+np.sin(m.pi*u.real/2)),c,d)

def _compose(function, *funcs):
    if funcs:
        return lambda u:_change_origin(*function)(_compose(*funcs)(u))
    else:
        return lambda u:_change_origin(*function)(u)


class ImageTransform:
    """Defines usefull methods to make conformal mappings over an image.
    The input image is put between -1j and 1j with origin in the middle of the
    image (real part depends on the shape of the input).
    The output image is also put between -1j and 1j with origin in the middle 
    of the image (real part depends on the shape of the output)."""
    def __init__(self,name,suffix=0,output_width=1024,output_height=704,r=1.,c=1.,d=0.,blur=False,data=None):
        """Creates an object either from a file if data==None or from a numpy
        array given in data.
        name is the name of the input file and will be suffixed by suffix to
        get the name of the output file.
        d allows to translate the output.
        c scales and rotate the output before translation.
        r scales and rotate the output after translation.
        blur allows to blur the output.
        data allows to give the input as a numpy array instead of a file.        
        """
        self.name = somme(name.split('.')[0:-1])
        self.format = name.split('.')[-1]
        if data!=None:
            self.data = data
        else:
            self.data = plt.imread(name)
        self.suffix = suffix
        self._input_scaling = self.data.shape[1]/self.data.shape[0]
        self._width = output_width
        self._height = output_height
        self._output_scaling = output_width/output_height
        self._output_rect = [-self._output_scaling-1.j,self._output_scaling+1.j]
        self.c = r*c
        self.d = r*(d.real*self._output_scaling+1j*d.imag)
        self.blur = blur
        self._transformations=[]
    def mirror(self,X=2,Y=2,nbpix=0,color=[255,255,255]):
        """Replaces the input image by a mirrored version got by mirroring
        the input d along its right edge if X=2 and along its bottom edge if Y=2.
        Puts stripes with the number "nbpix" of pixels around the output and
        between the mirrored inputs with color "color"."""
        if self.format.upper()=='PNG':
            color = [color[k]/255 for k in range(3)]
        self.data = mirror(self.data,X=X,Y=Y,nbpix=nbpix,color=color)
        self._input_scaling = self.data.shape[1]/self.data.shape[0]
    def _f(self,z):
        u=(z+self.d)/self.c       
        if self._transformations:
            return _compose(*self._transformations)(u)
        else:
            return u
    def nouvtrans(self,fonc,c,d):
        """Add a transformation to be computed."""
        self._transformations+=[(fonc,c,d)]
        
    def similitude(self,c=1.,d=0.,auto=True):
        """Mapping z->c*(z-d)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(lambda u:u,c,d)
        
    def cut(self,rect=[-1.-1.j,1.+1.j],c=1.,d=0.,auto=True):
        """Fill the plane with copies of the given rectangle rect.
        If auto==True, the rectangle is the output image seen before the cut
        and complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            rect=[-self._output_scaling-1.j,self._output_scaling+1.j]
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(lambda u:rect[0]+(u.real-rect[0].real)%(rect[1].real-rect[0].real)+1.j*((u.imag-rect[0].imag)%(rect[1].imag-rect[0].imag)),c,d)
    
    def power(self,exponent,c=1.,d=0.,auto=True):
        """Mapping z->(c*(z-d))**exponent
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_power(exponent,c,d))
        
    def polsec(self,a,b,c=1.,d=0.,auto=True):
        """Mapping z->(c*(z-d))**2+a*c*(z-d)+b
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_second_degree_polynomial(a,b,c,d))
    
    def invpol(self,l,c=1.,d=0.,auto=True):
        """Reverse mapping of z->sum(l[i]*(c*(z-d))**i)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_inverse_polynomial(l,c,d))
    
    def exp(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->exp(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=m.pi/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_exponential(forma,N,P,Q,angle,c,d))
    def sin(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->sinh(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=m.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_sine(forma,N,P,Q,angle,c,d))
            
    def tan(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->tanh(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=m.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_tangent(forma,N,P,Q,angle,c,d))
    
    def symmetry4(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """A mapping built on tangent function with symmetry 4.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=m.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_symmetry4(forma,N,P,Q,angle,c,d))
    
    def symmetry4_v2(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """A mapping built on tangent function with symmetry 4.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=m.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_symmetry4v2(forma,N,P,Q,angle,c,d))
    
    def symmetry3(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """A mapping built on tangent function with symmetry 3.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=m.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_symmetry3(forma,N,P,Q,angle,c,d))
    
    def arcsin(self,auto=True,c=1.,d=0.):
        """Mapping z->asin(z)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        Furthermore, the output is then reduced by a factor pi/2.
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_complex_arcsine(c,d))
        if auto:
            self.similitude(c=2/m.pi)
        
    def arctan(self,auto=True,c=1.,d=0.):
        """Mapping z->atan(z)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        Furthermore, the output is then reduced by a factor pi/2.
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_complex_arctan(c,d))
        if auto:
            self.similitude(c=2/m.pi)
        
    def ln(self,auto=True,c=1.,d=0.):
        """Mapping z->ln(z)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        Furthermore, the output is then reduced by a factor pi.
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_complex_logarithm(c,d))
        if auto:
            self.similitude(c=1/m.pi)
        
#    def gd(self,Q=0):
#        self.tanComp(Angle=False,c=2,Q=Q)
#        self.lnComp()
        
    def oval(self,N=1,P=1,Q=0,auto=True,d=0.):  
        """A mapping built on sine function removing the corners of the input.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input.
        """
        if self._transformations==[]:
            forma=self._input_scaling
        else:
            forma=self._output_scaling
        if auto:
            d=d.real*forma+1j*d.imag
        self.nouvtrans(*_oval(forma,N,P,Q,d))
        if auto:
            self.similitude(c=(-1-1.j)/2**0.5)
    def fisheye(self,form=1.,c=1.,d=0.,auto=True):
        """A mapping built on sine function simulating fisheye.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
            * form is the scaling (width/height) of the transformation input.
        """
        if auto:
            if self._transformations==[]:
                form=self._input_scaling
            else:
                form=self._output_scaling
            d=d.real*form+1j*d.imag
        self.nouvtrans(*_fisheye(form,c,d))
    
    def invers_fisheye(self,form=1.,c=1.,d=0.,auto=True):
        """A mapping built on sine function simulating invers fisheye.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
            * form is the scaling (width/height) of the transformation input.
        """
        if auto:
            if self._transformations==[]:
                form=self._input_scaling
            else:
                form=self._output_scaling
            d=d.real*form+1j*d.imag
        self.nouvtrans(*_invers_fisheye(form,c,d))
        
    def equirectangular(self,c=1.,d=0.,auto=True):
        """Equirectangular projection allowing 3D panoramas of the whole
        complex plane.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:        
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag-1j/c
            else:
                d=d.real*self._output_scaling+1j*d.imag-1j/c
        self.nouvtrans(*_equirectangular(c,d))
    
    def _barycentre(self,indices):
            iu,iv=indices.real,indices.imag
            if self.blur:
                u,v=iu%self.data.shape[1],iv%self.data.shape[0]
                uu,vv=np.floor(u),np.floor(v)
                du,dv=u-uu,v-vv
                coef=np.array([(du if i else 1-du)*(dv if j else 1-dv) for i in range(2) for j in range(2)])
                if self.format.upper()=='PNG':                    
                    coef=np.array(list(zip(coef,coef,coef))).reshape(4,self._height,self._width,3)
                    try:
                        cl=coef[0]*self.data[[vv],[uu]][0]+coef[1]*self.data[[(vv+1)%self.data.shape[0]],[uu]][0]+coef[2]*self.data[[vv],[(uu+1)%self.data.shape[1]]][0]+coef[3]*self.data[[(vv+1)%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0]
                    except Exception:
                        print("Strange error :",str(self._transformations))
                        cl=coef[0]*self.data[[vv%self.data.shape[0]],[uu%self.data.shape[1]]][0]+coef[1]*self.data[[(vv+1)%self.data.shape[0]],[uu%self.data.shape[1]]][0]+coef[2]*self.data[[vv%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0]+coef[3]*self.data[[(vv+1)%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0]
                else:
                    coef=np.array(list(zip(coef,coef,coef))).reshape(4,self._height,self._width,3)
                    try:
                        cl=np.uint8(coef[0]*self.data[[vv],[uu]][0]+coef[1]*self.data[[(vv+1)%self.data.shape[0]],[uu]][0]+coef[2]*self.data[[vv],[(uu+1)%self.data.shape[1]]][0]+coef[3]*self.data[[(vv+1)%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0])
                    except Exception:
                        print("Strange error :",str(self._transformations))
                        cl=np.uint8(coef[0]*self.data[[vv%self.data.shape[0]],[uu%self.data.shape[1]]][0]+coef[1]*self.data[[(vv+1)%self.data.shape[0]],[uu%self.data.shape[1]]][0]+coef[2]*self.data[[vv%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0]+coef[3]*self.data[[(vv+1)%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0])
                return cl
            else:
                try:
                    cl=self.data[[np.floor(iv%self.data.shape[0])],[np.floor(iu%self.data.shape[1])],:]
                except Exception:
                    print("Strange error :",str(self._transformations))
                    cl=self.data[[np.floor(iv)%self.data.shape[0]],[np.floor(iu)%self.data.shape[1]],:]
                return cl[0]    
    
    def _trouveCouleurs(self,infini=True,liste=[[0,0]],MaxX=np.NaN,color=[255]*3):
        bx,by,ex,ey=self._output_rect[0].real,self._output_rect[0].imag,self._output_rect[1].real,self._output_rect[1].imag
        grid=np.mgrid[by:ey:(self._height*1j),bx:ex:(self._width*1j)]
        grid=(grid[0]*1j+grid[1]).conjugate()
        FPt=((self._f(grid)+(self._input_scaling+1j)*0.999)/2).conjugate()*self.data.shape[0]
        res=self._barycentre(FPt)
        if not(np.isnan(MaxX)):
            indices=np.array([[m.floor(j.real/self.data.shape[1])>MaxX for j in i] for i in FPt])
            res[indices]=color
        if infini:
            return res
        if self.format.upper()=='PNG':
            color=[color[k]/255 for k in range(3)]+[1.]
        else:
            color=np.uint8(color)
        indices=np.array([[[m.floor(j.real/self.data.shape[1]),m.floor(j.imag/self.data.shape[0])] not in liste for j in i] for i in FPt])
        res[indices]=color
        
        return res
    def transform(self,infinite=True,lst=[[0,0]],MaxX=np.NaN,color=[255]*3,print_and_save=True):
        """Perform the transformation(s).
        If infinite=True, the whole plane is covered by the input image or half
        the plane is MaxX is given.
        Else, the input is placed on places given in lst.
        If print_and_save=True, the ouput is automatically printed and saved.
        In any case, returns the array containing the output.
        """
        start = time.time()
        Couleurs=self._trouveCouleurs(infini=infinite,liste=lst,color=color,MaxX=MaxX)
        print("Calcul :",time.time()-start)
        if print_and_save:
            plt.imshow(list(Couleurs), cmap=plt.cm.gray)
            plt.imsave(self.name+"-"+str(self.suffix)+'.'+self.format,Couleurs)
        return Couleurs
    def video(self,trans,nbim,filename,gif=True):
        """Makes a gif or a video from an input image applying transformations
        given in trans parameter which evolve according to the integer value of
        a variable called i.
        trans must be a string containing the transformations separated by ;
        nbim is the number of images of the output.
        filename is the name of the file saved.
        If gif is True, the output is a GIF.
        """
        imgs=[]
        s=trans.split(";")
        for i in range(nbim):
            for t in s:
                exec("self."+t)
            start = time.time()
            Couleurs=self._trouveCouleurs()
            print("Image "+str(i)+" - Calcul :",time.time()-start)
            self._transformations=[]
            imgs.append(Couleurs)
        v=vided.ImageSequenceClip(imgs,10,with_mask=False)
        if gif:
            v.to_gif(filename,fps=10,loop=0,program='ImageMagick')
        else:
            v.to_videofile(filename,fps=10,audio=False)
    def sample(self,rep='./'):
        trans={".invers_fisheye()",".fisheye()",".oval()",".ln()",".arctan()"
               ,".arcsin()",".symmetry4()",".symmetry4_v2()",".symmetry3()"
               ,".tan()",".sin()",".exp()",".tan(angle=False)",".sin(angle=False)"
               ,".exp(angle=False)",".symmetry4(angle=False)",".symmetry4_v2(angle=False)"
               ,".symmetry3(angle=False)",".invpol([0.,-1,1.5,-0.5])",".power(2,c=0.3,d=-1-1j)"
               ,".power(1.33,c=0.3-0.3j,d=-1-1j)",".power(0.5,d=-1-1j)"}
        for k in trans:
            exec("self"+k)
            start = time.time()
            Couleurs=self._trouveCouleurs()
            print(k+" - Calcul :",time.time()-start)
            plt.imsave(rep+self.name+k+'.'+self.format,Couleurs)
            self._transformations=[]

class VideoTransform(ImageTransform):
    def __init__(self,name,suffix=0,output_width=1024,output_height=704,r=1.,c=1.,d=0.,blur=False):
        self.video=vided.VideoFileClip(name,verbose=True)
        self.duration=self.video.duration
        self.numpict=m.ceil(self.video.duration*self.video.fps)
        self.mirrored=False
        ImageTransform.__init__(self,name,suffix,output_width,output_height,r,c,d,blur,data=self.video.get_frame(0))
    def mirror(self,X=2,Y=2,nbpix=0,color=[255,255,255]):
        self.mirrored=True
        self.X=X
        self.Y=Y
        self.nbpix=nbpix
        self.color=color
    def transform(self,infinite=True,lst=[[0,0]],MaxX=np.NaN,color=[255]*3): 
        imgs=[]
        for i in range(self.numpict):
            start = time.time()    
            if self.mirrored:
                ImageTransform.mirror(self,self.X,self.Y,self.nbpix,self.color)
            imgs.append(ImageTransform.transform(self,print_and_save=False,infinite=infinite,lst=lst,MaxX=MaxX,color=color))
            print("Image "+str(i)+" - Calcul :",time.time()-start)
            self.data=self.video.get_frame(i*self.duration/self.numpict)
        v=vided.ImageSequenceClip(imgs,self.video.fps,with_mask=False)
        v=v.set_audio(self.video.audio)
        v.to_videofile(self.name+'-'+str(self.suffix)+'.avi',fps=self.video.fps)

# Standard tansformation : 
#T=ImageTransform("yourfile.jpg","out"
#                ,r=0.7*(1.-0.j),c=1.*(1.-0.j)
#                ,d=0.+0.j,output_width=1024
#                ,output_height=704,blur=False)
#T.mirror(X=2,Y=2,nbpix=10,color=[200,180,10])
#T.sin()
#T.transform()

# Video from a picture : 
#T=ImageTransform("yourfile.jpg",0
#                ,r=1.*(1.-0.j),c=1.*(1.-0.j)
#                ,d=0.+0.j,output_width=1024
#                ,output_height=704,blur=False)
#T.mirror(X=2,Y=2,nbpix=10,color=[200,180,10])
#T.video("symmetry3(angle=True,N=1,P=1,Q=0,d=0.2+0.04j*i)",50
#        ,"./Sentinel-out.gif",gif=True)

# Sample of implemented transformations : 
#T=ImageTransform("yourfile.jpg","out"
#                ,r=0.7*(1.-0.j),c=1.*(1.-0.j)
#                ,d=0.+0.j,output_width=1024
#                ,output_height=704,blur=False)
#T.mirror(X=1,Y=1,nbpix=10,color=[200,180,10])
#T.sample("./Sample/")

# Transformations on videos :
#V=VideoTransform("yourfile.mp4",0,r=1.*(1.-0.j),c=1.*(1.-0.j),d=0.+0.j
#                ,output_width=720,output_height=405,blur=False)
#V.mirror(X=1,Y=1,nbpix=0,color=[255,255,255])
#V.tan(angle=False,N=2)
#V.transform()
