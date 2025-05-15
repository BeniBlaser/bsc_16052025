import string

def set_plot_params():
    plot_params = dict()
    plot_params['fontsize'] = 16
    plot_params['fontsize_title'] = plot_params['fontsize']+2
    plot_params['fontsize_suptitle'] = plot_params['fontsize']+4
    plot_params['fontsize_ticklabels'] = plot_params['fontsize']-3
    plot_params['fontsize_annotation'] = plot_params['fontsize']
    plot_params['fontsize_panelstrings'] = plot_params['fontsize']+5
    return plot_params


def setup_map_figure_for_roms_romsoc_present_future_comparison(params,topo_lon_shuffle,topo_lat,topo_shuffle,add_cbax_for_all=True,regional_zoom='none',coupling_effect=False):
    configs = params.model_configs
    if coupling_effect == True:
        configs = configs + [params.model_configs[1]+' - '+params.model_configs[0]]
    plot_params = set_plot_params()
    plt.rcParams['font.size']= plot_params['fontsize']
    fig, ax = plt.subplots(len(configs),len(params.scenarios),figsize=(15,14),sharex=True,sharey=True)
    panelstrings = list(string.ascii_lowercase)
    panel_index = 0
    for cdx,config in enumerate(configs):
        print(config)
        #if config == 'romsoc_fully_coupled':
        #    config = 'romsoc_fc'
        #elif config == 'romsoc_fully_coupled - roms_only':
        #    config = 'romsoc_fc - roms'
        config_label_y_positions = [0.83,0.51,0.18]
        fig.text(0.04,config_label_y_positions[cdx],config.replace("_"," ").upper().replace(' ','\n'),ha='center',va='center',fontweight='bold')
        scenario_label_x_positions = [0.27,0.53,0.79]
        for sdx,scenario in enumerate(params.scenarios):
            if cdx == 0:
                fig.text(scenario_label_x_positions[sdx],0.98,scenario.upper(),ha='center',va='center',fontweight='bold')
            #ax[cdx,sdx].set_title('{}, {}'.format(config.replace("_"," ").upper(),scenario))
            ax[cdx,sdx].contourf(topo_lon_shuffle,topo_lat,topo_shuffle,colors='#555555',linewidths=2)
            if regional_zoom == 'none':
                ax[cdx,sdx].set_xlim([110,293])
                ax[cdx,sdx].set_ylim([-80,70])
            elif regional_zoom == 'CalCS':
                ax[cdx,sdx].set_xlim([220,255])
                ax[cdx,sdx].set_ylim([20,55])           
            elif regional_zoom == 'Eastern_Pacific':
                ax[cdx,sdx].set_xlim([200,292])
                ax[cdx,sdx].set_ylim([-20,62])   
            # set panel label and update panel index
            ax[cdx,sdx].text(0.96,0.96,panelstrings[panel_index]+')',ha="right", va="top",bbox=dict(boxstyle="round",ec='k',fc='w'),transform=ax[cdx,sdx].transAxes,fontsize=plot_params['fontsize_panelstrings'])
            panel_index += 1   
    plt.tight_layout()
    plt.subplots_adjust(left=0.14,top=0.96)        
    if add_cbax_for_all == True:
        plt.subplots_adjust(right=0.9,wspace=0.1,hspace=0.1)
        cbax = fig.add_axes([0.92,0.2,0.02,0.6])
        return fig, ax, configs, cbax
    else:
        return fig, ax, configs
    
def finalize_axes(ax):
    plot_params = set_plot_params()
    axshape = np.shape(ax)
    for rowdx in range(axshape[0]):
        for coldx in range(axshape[1]):
            ax[rowdx,coldx] = update_longitude_labels_on_xaxis(ax[rowdx,coldx],plot_params,label='off')
            ax[rowdx,coldx] = update_latitude_labels_on_yaxis(ax[rowdx,coldx],plot_params,label='off')
    return ax

def axi_contourf(params,ax,lon,lat,mask,data_dict,configs):
    """ 
    data_dict musk be a dictionary with the following shape:
    data_dict['config']['scenario']
    """
    axshape = np.shape(ax)
    assert(axshape[0]==len(data_dict.key()))
    assert(axshape[1]==len(params.scenarios))
    for rowdx,config in enumerate(configs):
        for coldx,scenario in enumerate(params.scenarios):
            data = data_dict[config][scenario]
            ax[rowdx,coldx].contourf(lon,lat,data*mask)
    return ax
    

def update_longitude_labels_on_xaxis(ax,plot_params,label='on'):
    xticks = ax.get_xticks()
    xticks2 = np.abs(360-xticks)
    xticklabs = []
    for it,tick in enumerate(xticks):
        tickvals = [xticks[it],xticks2[it]]
        if np.argmin(tickvals)==0:
            WorE = 'E'
        elif np.argmin(tickvals)==1:
            WorE = 'W'
        minval = np.min(tickvals)
        xticklabs.append("{:2d}°{}".format(int(minval),WorE))
    ax.set_xticklabels(xticklabs,fontsize=plot_params['fontsize_ticklabels'])
    if label == 'on':
        ax.set_xlabel('Longitude')
    else:
        ax.set_xlabel('')
    return ax

def update_longitude_labels_on_yaxis(ax,plot_params,label='on'):
    yticks = ax.get_yticks()
    yticks2 = np.abs(360-yticks)
    yticklabs = []
    for it,tick in enumerate(yticks):
        tickvals = [yticks[it],yticks2[it]]
        if np.argmin(tickvals)==0:
            WorE = 'E'
        elif np.argmin(tickvals)==1:
            WorE = 'W'
        minval = np.min(tickvals)
        yticklabs.append("{:2d}°{}".format(int(minval),WorE))
    ax.set_yticklabels(yticklabs,fontsize=plot_params['fontsize_ticklabels'])
    if label == 'on':
        ax.set_ylabel('Longitude')
    else:
        ax.set_ylabel('')
    return ax

def update_latitude_labels_on_xaxis(ax,plot_params,label='on'):
    xticks = ax.get_xticks()
    xticklabs = []
    for it,tick in enumerate(xticks):
        tick = int(tick)
        if tick==0:
            NorS = ''
        elif tick < 0:
            NorS = 'S'
        elif tick > 0:
            NorS = 'N'
        xticklabs.append("{:2d}°{}".format(int(np.abs(tick)),NorS))
    ax.set_xticklabels(xticklabs,fontsize=plot_params['fontsize_ticklabels'])
    if label == 'on':
        ax.set_xlabel('Latitude')
    else:
        ax.set_xlabel('') 
    return ax
    
def update_latitude_labels_on_yaxis(axi,plot_params,label='on'):
    yticks = axi.get_yticks()
    yticklabs = []
    for it,tick in enumerate(yticks):
        tick = int(tick)
        if tick==0:
            NorS = ''
        elif tick < 0:
            NorS = 'S'
        elif tick > 0:
            NorS = 'N'
        yticklabs.append("{:2d}°{}".format(int(np.abs(tick)),NorS))
    axi.set_yticklabels(yticklabs,fontsize=plot_params['fontsize_ticklabels'])
    if label == 'on':
        axi.set_ylabel('Latitude')
    else:
        axi.set_ylabel('')    
    return axi
