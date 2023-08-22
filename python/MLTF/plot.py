# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

import logging
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import random
import os
from . import calc

logger = logging.getLogger(__name__)

# plot function for history
def plot_history_ax(ax, history, ftsize=18, xscalelog=False, yscalelog=False, label='training set'):
    ax.plot(history, ms=2, lw=1, label=label)
    ax.axhline(0.0, color="black", lw=2)
    if xscalelog:
        ax.set_xscale('log')
    if yscalelog:
        ax.set_yscale('log')
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch', fontsize=ftsize)
    ax.set_ylabel('Loss value', fontsize=ftsize)
    ax.legend(loc='best', prop={'size': ftsize-6})
    
    

def plot_history(history, filename):
    if ( (type(history)!=list) | (type(history)!=np.array)):
        try:
            with open(history) as f:
                history = json.load(f)
        except OSError:
            logger.info("unable to read %s"%(history))    
            raise
    plt.plot(history, ms=2, lw=1)
    plt.axhline(0.0, color="black", lw=2)

    plt.ylim(0.5*min(history), 1.5*max(history))
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.tight_layout()
    plt.savefig(filename,dpi=200)
    plt.close()

def plot_bias(targets, meanpreds, msbval=None, ylabel='ylabel', filename=None):
    if msbval is not None:
        label="min MSB: %.3e"%(msbval)
    else:
        label=None
    plt.plot(targets, meanpreds, marker=".", color="#d95f02", label=label, ls="None", ms=2)
    plt.axhline(0.0, color="black", lw=1)
    plt.xlabel(r"$g_{1}$", fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    if label is not None:
        plt.legend(fontsize=20, markerscale=4, numpoints=1)
    #plt.xlim(0.15, 2.1)
    #plt.ylim(-0.01, 0.01)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename,dpi=200)
    plt.close()

  

def color_plot(x,y,z,colorlog,xtitle,ytitle,ctitle,title=None, ftsize=16,xlim=None, ylim=None,cmap=None,filename=None, npoints_plot=None, linreg=False, xscalelog=False, yscalelog=False, yerr=None, s=4.0, alpha=0.9, alpha_err=0.1, sidehists=False, sidehistxkwargs={}, sidehistykwargs={}):
    from matplotlib.ticker import StrMethodFormatter, ScalarFormatter

    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,2.f}'))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.clf()
  
    if (npoints_plot is None): npoints_plot=len(x)
    if (npoints_plot> len(x)): npoints_plot=len(x)
    inds=random.sample([i for i in range(len(x))],npoints_plot)
    logger.info("using %i point for the plot"%(npoints_plot))
    if linreg:
        logger.info("Doing a fast regresion in color plot")
        xplot=np.linspace(min(x),max(x))
        #mask=~x.mask&~y.mask
        #x_unmask,y_unmask=x[mask].data,y[mask].data
        are_maskx=type(x)==np.ma.masked_array
        are_masky=type(y)==np.ma.masked_array
        are_masked=(np.ma.is_masked(x))|(np.ma.is_masked(y))|(are_maskx|are_masky)
        if are_masked:
            mask=~x.mask&~y.mask
            x_unmask,y_unmask=x[mask].data,y[mask].data
        else:
            logger.info("inputs are not masked arrays")
            x_unmask,y_unmask=x,y
        
        ret=calc.linreg(x_unmask,y_unmask)
        m,merr,c, cerr=(ret["m"]+1),ret["merr"],ret["c"],ret["cerr"]
        plt.plot(xplot,m*xplot+c, ls='-',linewidth=2, color='red', label='$\mu$: %.4f $\pm$ %.4f \n  c: %.4f $\pm$ %.4f'%(m,merr,c, cerr ))
        plt.legend(loc='best', prop={'size': ftsize-6})

    if colorlog: 
        z=abs(z)
        colornorm=LogNorm( vmin=np.nanmin(z), vmax=np.nanmax(z))
        #colornorm=LogNorm( vmin=1.e-3, vmax=1.e-0)
    else: colornorm=None
    if z is not None:
        if yerr is not None:
            ebarskwargs = {"fmt":'none', "color":"black", "ls":":",'elinewidth':0.5, 'alpha':alpha_err}
            plt.errorbar(x[inds], y[inds], yerr=yerr, **ebarskwargs)
        sct=plt.scatter(x[inds], y[inds],c=z[inds], norm=colornorm, marker=".",alpha=alpha,cmap=cmap,s=s)
    else:
        if yerr is not None:
            ebarskwargs = {"fmt":'none', "color":"black", "ls":":", 'elinewidth':0.5, 'alpha':alpha_err}
            plt.errorbar(x[inds], y[inds], yerr=yerr, **ebarskwargs)
        sct=plt.scatter(x[inds], y[inds], norm=colornorm, marker=".",alpha=alpha,cmap=cmap,s=s)

    plt.xlabel(xtitle, fontsize=ftsize)
    plt.ylabel(ytitle, fontsize=ftsize)
    if title is not None: plt.title(title, fontsize=ftsize-4, loc='left', color='red')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if xscalelog:
        plt.xscale('log')
    if yscalelog:
        if np.any(np.array(y[inds])!=0):
            plt.yscale('log')

    if z is not None:
        cbar=plt.colorbar(sct)
        #cbar.ax.set_xlabel(ctitle, fontsize=ftsize-2)
        #cbar.ax.xaxis.set_label_coords(0.5,1.1)
        cbar.ax.set_ylabel(ctitle, fontsize=ftsize-2, rotation=-90)
        cbar.ax.yaxis.set_label_coords(5.5,0.5)
    if sidehists:
        ax=plt.gca()
        histxkwargs = {"histtype":"stepfilled", "bins":100, "ec":"none", "color":"gray", "range":None, "log":False}
        histxkwargs.update(sidehistxkwargs)
        histykwargs = {"histtype":"stepfilled", "bins":100, "ec":"none", "color":"gray", "range":None, "log":False}
        histykwargs.update(sidehistykwargs)

        divider = make_axes_locatable(ax)
        axhistx = divider.append_axes("top", 1.0, pad=0.15, sharex=ax)
        axhisty = divider.append_axes("right", 1.0, pad=0.15, sharey=ax)

        axhistx.hist(x, **histxkwargs)
        axhisty.hist(y, orientation='horizontal', **histykwargs)
        for tl in axhistx.get_xticklabels():
            tl.set_visible(False)
        for tl in axhisty.get_yticklabels():
            tl.set_visible(False)
        
        axhistx.ticklabel_format(style='sci', axis="y", scilimits=(0,0), useMathText=True )
        #axhistx.yaxis.tick_right()

        axhisty.ticklabel_format(style='sci', axis="x", scilimits=(0,0), useMathText=True )
        axhisty.xaxis.tick_top()
        axhisty.tick_params(axis='x', labelrotation=-90)
        ax_max = max(axhisty.get_xticks())
        axhisty.get_xaxis().get_offset_text().set_visible(False)
        exponent_axis = np.floor(np.log10(ax_max)).astype(int)
        axhisty.annotate(r'$\times$10$^{%i}$'%(exponent_axis), xy=(1.01, .89), xycoords='axes fraction', rotation=-90)

        #axhisty.get_xaxis().get_offset_text().set_rotation(-90)
        #axhisty.get_xaxis().get_offset_text().set_position((1.2,0))

        if title:
            axhistx.set_title(title)
            
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename,dpi=200)

    plt.close()

def color_plot_ax(ax,x,y,c,colorlog,xtitle,ytitle,ctitle,ftsize,xlim=None, ylim=None,cmap=None,filename=None):
    #plt.clf()
    if colorlog: 
        c=abs(c)
        colornorm=LogNorm( vmin=np.nanmin(c), vmax=np.nanmax(c))
    else: colornorm=None
    sct=ax.scatter(x, y,c=c, norm=colornorm, marker=".",alpha=0.7,cmap=cmap)
    ax.set_xlabel(xtitle, fontsize=ftsize)
    ax.set_ylabel(ytitle, fontsize=ftsize)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    cbar=plt.colorbar(sct, ax=ax)
    cbar.ax.set_xlabel(ctitle, fontsize=ftsize-2)
    cbar.ax.xaxis.set_label_coords(0.5,1.1)
    #plt.tight_layout()
    if filename is not None:
        plt.savefig(filename,dpi=200)
    

def plot_correlation_matrix(mx, my, nbins=10, filename=None):
    #mx is g1, my is w1
    #makind 2d catalog
    sel=~mx.mask&~my.mask
    mx=mx[sel]
    my=my[sel]
    print(mx)
    
    joincat=np.ma.masked_array(np.vstack((mx.data,my.data)).T, mask=np.vstack((mx.mask,my.mask)).T )
    #sorting ascending in mx
    sorted_join=np.sort(joincat.T, axis=1)
    #to use np.cov with need equal size samples
    binsize=int(len(sorted_join.T)/nbins)
    indx=[binsize*(i+1) for i in range(nbins)] 
    splited_join=np.array_split(sorted_join, nbins, axis=1)[:nbins]
    mxlist=[ ma[0] for ma in splited_join]
    mylist=[ ma[1] for ma in splited_join]

    print(mxlist)
    covmat=np.ma.cov(mxlist+mylist)
    plt.clf()
    #lengths = [binsize*nbins]*2
    lengths = [1]*nbins*2

    #plt.imshow(covmat,cmap='viridis'+'_r', interpolation='nearest', aspect='auto')
    #plt.colorbar()
    plotcorrmat(covmat)
    
    

    pos_lines = [-0.5]
    for i in range(len(lengths)):
        pos_lines.append(pos_lines[i] + lengths[i])
    pos_lines = pos_lines[1:-1]
    for line in pos_lines:
        plt.axvline(x=line, c='k', lw=1, ls='-')
        plt.axhline(y=line, c='k', lw=1, ls='-')
    plt.tight_layout()
    
    plt.savefig(filename, dpi=200)
    plt.close()
    
def plotcorrmat(cov):
    #cov = np.mat(cov)
    D = np.ma.diag(np.sqrt(np.ma.diag(cov)))
    d = np.linalg.inv(D)
    corr = d*cov*d
    print(corr)
    cov_vmin=np.ma.min(corr)
    plt.imshow(corr,cmap='viridis'+'_r', interpolation='nearest',
               aspect='auto', origin='lower', vmin=cov_vmin, vmax=1.)
    plt.colorbar()
