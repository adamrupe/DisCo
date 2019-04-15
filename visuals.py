'''
Code for visualizations of 1+1 D and 2+1 D systems and analytics.

'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import animation, rc
from matplotlib.colors import LinearSegmentedColormap

###################################################################################################

def diagram(field, ax=None, size=16, colorbar=False, y_min=0, y_max=None,
                x_min=0, x_max=None, ticks=True, xtick_spacing=10, ytick_spacing=10,
                tick_size=None, cmap=plt.cm.Greys, edgecolors='none', return_pcolor=False, invert_y=True,
                **pcolormesh_kwargs):
    '''
    Plots the given 2D field using matplotlib pcolormesh. Returns a matplotlib
    Axes object.

    Parameters
    ----------
    field: ndarray (2-D)
        2-D array of data to be plotted.
    ax: matplotlib Axes object, optional (default=None)
        An external Axes object that may be passed in to be further manipulated
        after calling this plotting function.
    size: int, optional (default=16)
        Sets the size of the Figure object.
    colorbar: bool, optional (default=False)
        Set to True to include a colorbar in the plot, False to not include a colorbar.
    y_min: int, optional (default=0)
        Lower limit of the y-axis to be plotted.
    y_max: int, optional (default=None)
        Upper limit of the y-axis to be plotted. If None, is set to the size of the
        y-axis for the input field.
    x_min: int, optional (default=0)
        Lower limit of the x-axis to be plotted.
    x_max: int, optional (default=None)
        Upper limit of the x-axis to be plotted. If None, is set to the size of the
        x-axis for the input field.
    ticks: bool, optional (default=None)
        Determines whether to display the axis tick marks and corresponding labels or not.
    xtick_spacing: int, optional (default=10)
        Sets the interval for ticks on along the x-axis.
    ytick_spacing: int, optional (default=10)
        Sets the interval for ticks along the y-axis.
    tick_size: int, optional (default=None)
        Sets the size of the tick labels for the axes. If None, defaults to the value
        of the 'size' parameter.
    cmap: matplotlib colormap, optional (default=plt.cm.Greys)
        Colormap used by pcolormesh for plotting the input field.
    edgecolors: valid matplotlib color, optional (default='black')
        Sets the color of the gird lines outlining the cells of the field.
        If set to 'none' does not display any grid lines.
    **pcolormesh_kwargs:
        Additional keyword arguments for the matplotlib pcolormesh plotting function.

    Returns
    -------
    ax: matplotlib Axes object
        Axes object that has either been passed in or created, then updated with
        this function.
    '''
    h,w = np.shape(field)
    if y_max is None:
        y_max = h
    if x_max is None:
        x_max = w
    H = y_max - y_min
    W = x_max - x_min
    if ax is None:
        fig, ax = plt.subplots(figsize = (size, (H/W)*size))

    cropped_field = field[y_min:y_max, x_min:x_max]
    im = ax.pcolormesh(cropped_field, cmap=cmap, edgecolors=edgecolors, **pcolormesh_kwargs)
    if invert_y:
        ax.invert_yaxis()

    if colorbar:
        fig.colorbar(im)

    # various code for tick control
    if ticks:
        ax.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
        if tick_size is None:
            tick_size = size
        x_labels = [str(int(label)+x_min+xtick_spacing) for label in ax.get_xticks().tolist()]
        y_labels = [str(int(label)+y_min+ytick_spacing) for label in ax.get_yticks().tolist()]
        x_ticks = [tick+0.5 for tick in ax.get_xticks()]
        y_ticks = [tick +0.5 for tick in ax.get_yticks()]
        if W % xtick_spacing == 0:
            x_trim = 2
        else:
            x_trim = 1
        if H % ytick_spacing == 0:
            y_trim = 2
        else:
            y_trim = 1
        ax.set_xticks(x_ticks[1:len(x_ticks)-x_trim], minor=True)
        ax.set_yticks(y_ticks[1:len(y_ticks)-y_trim], minor=True)
        ax.set_xticklabels(x_labels, fontsize=tick_size, minor=True)
        ax.set_yticklabels(y_labels, fontsize=tick_size, minor=True)
        ax.xaxis.set_major_formatter(plticker.NullFormatter())
        ax.yaxis.set_major_formatter(plticker.NullFormatter())
        ax.tick_params(
                axis='both',
                which='minor',
                direction='out',
                top='off',
                right='off',
                pad=8)
        ax.tick_params(
                axis='both',
                which='major',
                top='off',
                bottom='off',
                left='off',
                right='off')
    else:
        ax.tick_params(axis='both',
                        which='both',
                        bottom='off',
                        top='off',
                        left='off',
                        right='off',
                        labelleft='off',
                        labelbottom='off')
    if return_pcolor:
        return im
    else:
        return ax

###################################################################################################

def spacetime_diagram(field, ax=None, size=16, title=None, title_size=None, axes_label_size=None,
                      axes=True, t_min=0, t_max=None, x_min=0, x_max=None, ttick_spacing=10,
                      **kwargs):
    '''
    Plots the given 1+1 D spacetime field using matplotlib pcolormesh.
    Expects time on the vertical axis and space on the horizontal. Returns a matplotlib
    Axes object.

    Parameters
    ----------
    field: ndarray (2-D)
        2-D array of data to be plotted.
    ax: matplotlib Axes object, optional (default=None)
        An external Axes object that may be passed in to be further manipulated
        after calling this plotting function.
    size: int, optional (default=16)
        Sets the size of the Figure object.
    title: str, optional (default=None)
        Title for the figure.
    title_size: int, optional (default=None)
        Sets the size for the figure title. If None, defaults to 2 times the 'size'
        parameter.
    axes_label_size: int, optional (default=None)
        Sets the size for the labels on the x and y axes. If None, defaults to
        1.8 times the 'size' parameter.
    axes: bool, optional (default=True)
        Determines whether to display the axes, including labels and ticks.
    t_min: int, optional (default=0)
        Lower limit of the time (vertical) axis to be plotted.
    t_max: int, optional (default=None)
        Upper limit of the time (vertical) axis to be plotted. If None, is set to the size of the
        time axis for the input field.
    x_min: int, optional (default=0)
        Lower limit of the x-axis (horizontal) to be plotted.
    x_max: int, optional (default=None)
        Upper limit of the x-axis (horizontal) to be plotted. If None, is set to the size of the
        x-axis for the input field.
    ttick_spacing: int, optional (default=10)
        Sets the interval for ticks along the time axis.
    **kwargs:
        Additional keyword arguments for the 'diagram' plotting function that is used
        as the backend.

    Returns
    -------
    ax: matplotlib Axes object
        Axes object that has either been passed in or created, then updated with
        this function.
    '''
    if ax is None:
        h,w = np.shape(field)
        if t_max is None:
            t_max = h
        if x_max is None:
            x_max = w
        H = t_max - t_min
        W = x_max - x_min

        fig, ax = plt.subplots(figsize = (size, (H/W)*size))

    diagram(field, ax=ax, y_min=t_min, y_max=t_max, x_min=x_min, x_max=x_max,
            ytick_spacing=ttick_spacing, **kwargs)

    if not axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    if title is not None:
        if title_size is None:
            title_size = 2*size
        ax.set_title(title, {'fontsize':title_size})
    if axes_label_size is None:
        axes_label_size = 1.8*size
    ax.set_ylabel('Time', {'fontsize':axes_label_size})
    ax.set_xlabel('Space', {'fontsize':axes_label_size})

    return ax

###################################################################################################

def state_overlay_diagram(data_field, state_field, ax=None, size=16, field_colors=plt.cm.Greys,
                          text_colors=None, text_single_color=None, state_labels=None,
                          text_size=None, x_min=0, x_max=None, y_min=0, y_max=None, **kwargs):
    '''
    Plots the given 2D data field with text of values from the state field overlayed
    on top, using matplotlib pcolormesh. Returns a matplotlib Axes object.

    Parameters
    ----------
    data_field: array-like
        2-D array of data to be plotted.
    state_field: array-like
        2-D array of state field values to be overlayed as text on top of the
        data field plot. Must be same shape as data_field.
    ax: matplotlib Axes object, optional (default=None)
        An external Axes object that may be passed in to be further manipulated
        after calling this plotting function.
    size: int, optional (default=16)
        Sets the size of the Figure object.
    field_colors: matplotlib colormap, optional (default=plt.cm.Greys)
        Colormap for the data_field values
    text_colors: matplotlib colormap, optional (default=None)
        Colormap for the overlaid state labels. If left as None, will color all
        labels the same according to text_single_color (default=blue).
    text_single_color: matplotlib color, optional (default=None)
        If text_colors left as None, all labels will be colored uniformly as the
        color specified here. If left as None, default is a shade of blue given
        by plt.cm.rainbow(1/5). text_colors takes precedence, so that if both
        text_colors and text_single_color are not None, colors will be assigned
        according to the text_colors colormap.
    state_labels: dict, optional (default=None)
        Dictionary that maps the numerical state values in state_field to custom
        labels given in the dictionary.
    text_size: int, optional (default=None)
        Size of the overlayed label text. If left as None, will be given as the
        plot size (size parameter).
    y_min: int, optional (default=0)
        Lower limit of the y-axis to be plotted.
    y_max: int, optional (default=None)
        Upper limit of the y-axis to be plotted. If None, is set to the size of the
        y-axis for the input field.
    x_min: int, optional (default=0)
        Lower limit of the x-axis to be plotted.
    x_max: int, optional (default=None)
        Upper limit of the x-axis to be plotted. If None, is set to the size of the
        x-axis for the input field.
    **kwargs:
        Additional keyword arguments for the 'diagram' plotting function that is used
        as the backend.

    Returns
    -------
    ax: matplotlib Axes object
        Axes object that has either been passed in or created, then updated with
        this function.
    '''
    if data_field.shape != state_field.shape:
        raise ValueError("data_field and state_field must be the same shape.")

    h,w = np.shape(data_field)
    if y_max is None:
        y_max = h
    if x_max is None:
        x_max = w
    H = y_max - y_min
    W = x_max - x_min
    y_ind = np.arange(y_min, y_max, 1)
    x_ind = np.arange(x_min, x_max, 1)
    x, y = np.meshgrid(x_ind, y_ind)
    if ax is None:
        fig, ax = plt.subplots(figsize = (size, (H/W)*size))

    #diagram(data_field, ax=ax, cmap=field_colors, **kwargs)
    diagram(data_field, ax=ax, cmap=field_colors, y_min=y_min, y_max=y_max,
            x_min=x_min, x_max=x_max, **kwargs)

    #add state label text
    states = np.unique(state_field)
    max_state = np.max(states)
    if state_labels is None:
        state_labels = dict(zip(states,states))
    if text_colors is None:
        if text_single_color is None:
            text_single_color = plt.cm.rainbow(1/5)
        text_colors = {state:text_single_color for state in states}
    if text_size is None:
        text_size = size
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = state_field[y_val][x_val]
        if c not in state_labels:
            raise ValueError("Need a label for state {} in state_labels dict".format(c))
        if type(text_colors) is dict:
            if c not in text_colors:
                raise ValueError("Need a color for state {} in text_colors dict".format(c))
            else:
                text_color = text_colors[c]
        else:
            text_color = text_colors(c/max_state)
        ax.text(x_val+0.5-x_min, y_val+0.5-y_min, state_labels[c], va='center', ha='center',
                color=text_color, size=text_size, weight='black')

    return ax

###################################################################################################

def custom_colors(color_map):
    '''
    Creates custom color map from given dictionary.

    Parameters
    ----------
    color_map: dict
        Keys are field array values and dict values are the corresponding colors
        to be assigned to the field array values. Keys must be integers. Vals must
        be valid matplotlib color scheme.

    Returns
    -------
    color_map: LinearSegmentedColormap
        Returns matplotlib LinearSegmentedColormap.from_list using the given
        color_map dictionary.
    '''
    keys = np.array(color_map.keys())
    min_key = np.min(keys)
    normalization = np.max(keys - min_key)
    normalized_keys = (keys-min_key)/normalization
    normed_map = dict(zip(normalized_keys, color_map.values()))
    colors = LinearSegmentedColormap.from_list('mycmap', sorted(normed_map.items()))
    return colors

def old_custom_colors(values, colors):
    '''
    Creates custom color map from given list of values to the corresponding colors.
    That is, values[i] is mapped to color colors[i]/

    Parameters
    ----------
    values: list
        List of the numerical values to be given colors.
    colors: list
        List of colors to be assigned to the given values.

    Returns
    -------
    color_map: LinearSegmentedColormap
        Returns matplotlib LinearSegmentedColormap.from_list using the input values
        and colors
    '''
    min_val = np.min(values)
    normalization = np.max(values - min_val)
    normalized_vals = (values-min_val)/normalization
    color_map = LinearSegmentedColormap.from_list('mycmap', sorted(zip(normalized_vals, colors)))
    return color_map

###################################################################################################

def state_overlay_animate(data_fields, state_fields, ax=None, size=16, field_colors=plt.cm.Greys,
                          text_colors=None, text_single_color=None, state_labels=None,
                          text_size=None, edgecolors='black', axes_label_size=None,
                          y_min=0, y_max=None, x_min=0, x_max=None, colorbar=False,
                          title=None, title_size=None, axes=True, ticks=True,
                          ytick_spacing=10, xtick_spacing=10, tick_size=None, **anim_kwargs):
    '''
    Animates matplotlib pcolormesh plot of the given 2-D data fields with text of values
    from state fields overlayed on top. Returns a matplotlib animation object.

    Must be same number of state and data fields, and all fields (both data and
    state) must be same shape.

    Parameters
    ----------
    data_field: array-like
        2-D array of data to be plotted.
    state_field: array-like
        2-D array of state field values to be overlayed as text on top of the
        data field plot. Must be same shape as data_fields.
    size: int, optional (default=16)
        Sets the size of the Figure object.
    field_colors: matplotlib colormap, optional (default=plt.cm.Greys)
        Colormap for the data_field values
    text_colors: matplotlib colormap, optional (default=None)
        Colormap for the overlaid state labels. If left as None, will color all
        labels the same according to text_single_color (default=blue).
    text_single_color: matplotlib color, optional (default=None)
        If text_colors left as None, all labels will be colored uniformly as the
        color specified here. If left as None, default is a shade of blue given
        by plt.cm.rainbow(1/5). text_colors takes precedence, so that if both
        text_colors and text_single_color are not None, colors will be assigned
        according to the text_colors colormap.
    state_labels: dict, optional (default=None)
        Dictionary that maps the numerical state values in state_field to custom
        labels given in the dictionary.
    text_size: int, optional (default=None)
        Size of the overlayed label text. If left as None, will be given as the
        plot size (size parameter).
    edgecolors: matplotlib color, optional (default='black')
        Grid line colors for the diagram. Set to 'none' for no grid lines.
    axes_label_size: int, optional (default=None)
        Size of the axes labels "Time" and "Space". If left as None will be given
        as 1.8*size.
    y_min: int, optional (default=0)
        Lower limit of the y-axis to be plotted.
    y_max: int, optional (default=None)
        Upper limit of the y-axis to be plotted. If None, is set to the size of the
        y-axis for the input field.
    x_min: int, optional (default=0)
        Lower limit of the x-axis to be plotted.
    x_max: int, optional (default=None)
        Upper limit of the x-axis to be plotted. If None, is set to the size of the
        x-axis for the input field.
    colorbar: Bool, optional (default=False)
        Option to display the colorbar.
    title: str, optional (default=None)
        Title for the animated Figure.
    title_size: int, optional (default=None)
        Sets the size for the figure title. If None, defaults to 2 times the 'size'
        parameter.
    axes: bool, optional (default=True)
        Determines whether to show the axes, with ticks and labels or not.
    ticks: bool, optional (default=True)
        Determines whether to show the axes ticks or not. Does not affect axes labels.
    xtick_spacing: int, optional (default=10)
        Sets the interval for ticks on along the x-axis.
    ytick_spacing: int, optional (default=10)
        Sets the interval for ticks along the y-axis.
    tick_size: int, optional (default=None)
        Sets the size of the tick labels for the axes. If None, defaults to the value
        of the 'size' parameter.
    **anim_kwargs:
        Additional keyword arguments for the matplotlib animation.FuncAnimation()
        function.

    Returns
    -------
    ax: matplotlib Axes object
        Axes object that has either been passed in or created, then updated with
        this function.
    '''
    if np.shape(data_fields) != np.shape(state_fields):
        raise ValueError("data_fields and state_fields must be same shape.")

    t,h,w = np.shape(data_fields)
    if y_max is None:
        y_max = h
    if x_max is None:
        x_max = w
    H = y_max - y_min
    W = x_max - x_min
    if ax is None:
        fig, ax = plt.subplots(figsize = (size, (H/W)*size))

    cropped_data = data_fields[:, y_min:y_max, x_min:x_max]
    cropped_states = state_fields[:, y_min:y_max, x_min:x_max]

    im = ax.pcolormesh(cropped_data[0], cmap=cmap, edgecolors=edgecolors)

    ax.invert_yaxis()
    if not axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    if colorbar:
        ax.colorbar()
    if title is not None:
        if title_size is None:
            title_size = 2*size
        ax.set_title(title, {'fontsize':title_size})
    if axes_label_size is None:
        axes_label_size = 1.8*size
    ax.set_ylabel('Y', {'fontsize':axes_label_size})
    ax.set_xlabel('X', {'fontsize':axes_label_size})

    # various code for tick control
    if ticks:
        ax.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
        if tick_size is None:
            tick_size = size
        x_labels = [str(int(label)+x_min+xtick_spacing) for label in ax.get_xticks().tolist()]
        y_labels = [str(int(label)+y_min+ytick_spacing) for label in ax.get_yticks().tolist()]
        x_ticks = [tick+0.5 for tick in ax.get_xticks()]
        y_ticks = [tick +0.5 for tick in ax.get_yticks()]
        if W % xtick_spacing == 0:
            x_trim = 2
        else:
            x_trim = 1
        if H % ytick_spacing == 0:
            y_trim = 2
        else:
            y_trim = 1
        ax.set_xticks(x_ticks[1:len(x_ticks)-x_trim], minor=True)
        ax.set_yticks(y_ticks[1:len(y_ticks)-y_trim], minor=True)
        ax.set_xticklabels(x_labels, fontsize=tick_size, minor=True)
        ax.set_yticklabels(y_labels, fontsize=tick_size, minor=True)
        ax.xaxis.set_major_formatter(plticker.NullFormatter())
        ax.yaxis.set_major_formatter(plticker.NullFormatter())
        ax.tick_params(
                axis='both',
                which='minor',
                direction='out',
                top='off',
                right='off',
                pad=8)
        ax.tick_params(
                axis='both',
                which='major',
                top='off',
                bottom='off',
                left='off',
                right='off')
    else:
        ax.tick_params(axis='both',
                        which='both',
                        bottom='off',
                        top='off',
                        left='off',
                        right='off',
                        labelleft='off',
                        labelbottom='off')

    # set "global" text items to be used in all overlays
    states = np.unique(state_fields)
    max_state = np.max(states)
    if state_labels is None:
        state_labels = dict(zip(states,states))
    if text_colors is None:
        if text_single_color is None:
            text_single_color = plt.cm.rainbow(1/5)
        text_colors = {state:text_single_color for state in states}
    if text_size is None:
        text_size = size

    state_field = cropped_states[0]
    # each text instance needs to be assigned a viariable so it can be updated
    # during the animation
    labels = []
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = state_field[y_val][x_val]
        if c not in state_labels:
            raise ValueError("Need a label for state {} in state_labels dict".format(c))
        if type(text_colors) is dict:
            if c not in text_colors:
                raise ValueError("Need a color for state {} in text_colors dict".format(c))
            else:
                text_color = text_colors[c]
        else:
            text_color = text_colors(c/max_state)

        labels.append(ax.text(x_val+0.5, y_val+0.5, state_labels[c], va='center', ha='center',
                color=text_color, size=text_size, weight='black')
                     )

    plt.close() # prevents initial figure from being displayed in jupyter notebook

    # update function needed for animation
    def animate(i):
        im.set_array(cropped_data[i].flatten())
        state_field = cropped_states[i]
        for x_val, y_val, label in zip(x.flatten(), y.flatten(), labels):
            c = state_field[y_val][x_val]
            if c not in state_labels:
                raise ValueError("Need a label for state {} in state_labels dict".format(c))
            text_color = text_colors(c/max_state)
            label.set_text(state_labels[c])
            label.set_color(text_colors(c/max_state))

    frames = t

    # put it all together and call the matplotlib animation function
    anim = animation.FuncAnimation(fig, animate, frames=frames, **anim_kwargs)

    return anim


###################################################################################################
# 2+1 D STUFF
###################################################################################################


def config_diagram(field, ax=None, size=16, title=None, title_size=None, axes_label_size=None,
                      axes=True, y_min=0, y_max=None, x_min=0, x_max=None, **kwargs):
    '''
    Plots the given 2 D spatial field using matplotlib pcolormesh.
    Returns a matplotlib Axes object.

    Parameters
    ----------
    field: ndarray (2-D)
        2-D array of data to be plotted.
    ax: matplotlib Axes object, optional (default=None)
        An external Axes object that may be passed in to be further manipulated
        after calling this plotting function.
    size: int, optional (default=16)
        Sets the size of the Figure object.
    title: str, optional (default=None)
        Title for the figure.
    title_size: int, optional (default=None)
        Sets the size for the figure title. If None, defaults to 2 times the 'size'
        parameter.
    axes_label_size: int, optional (default=None)
        Sets the size for the labels on the x and y axes. If None, defaults to
        1.8 times the 'size' parameter.
    axes: bool, optional (default=True)
        Determines whether to display the axes, including labels and ticks.
    y_min: int, optional (default=0)
        Lower limit of the y-axis to be plotted.
    y_max: int, optional (default=None)
        Upper limit of the y-axis to be plotted. If None, is set to the size of the
        time axis for the input field.
    x_min: int, optional (default=0)
        Lower limit of the x-axis to be plotted.
    x_max: int, optional (default=None)
        Upper limit of the x-axis to be plotted. If None, is set to the size of the
        x-axis for the input field.
    **kwargs:
        Additional keyword arguments for the 'diagram' plotting function that is used
        as the backend.

    Returns
    -------
    ax: matplotlib Axes object
        Axes object that has either been passed in or created, then updated with
        this function.
    '''
    h,w = np.shape(field)
    if y_max is None:
        y_max = h
    if x_max is None:
        x_max = w
    H = y_max - y_min
    W = x_max - x_min
    if ax is None:
        fig, ax = plt.subplots(figsize = (size, (H/W)*size))

    diagram(field, ax=ax, y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max,
            **kwargs)

    if not axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    if title is not None:
        if title_size is None:
            title_size = 2*size
        ax.set_title(title, {'fontsize':title_size})
    if axes_label_size is None:
        axes_label_size = 1.8*size
    ax.set_ylabel('Y', {'fontsize':axes_label_size})
    ax.set_xlabel('X', {'fontsize':axes_label_size})

    return ax

###################################################################################################

def spacetime_animate(data_fields, ax=None, size=20, colorbar=False, axes_label_size=None,
                      y_min=0, y_max=None, x_min=0, x_max=None, title=None, invert_y=True,
                      title_size=None, axes=True, ticks=True, xtick_spacing=10,
                      ytick_spacing=10, tick_size=None, cmap=plt.cm.rainbow,
                      edgecolors='none', anim_kwargs={}, **pcolor_kwargs):
    '''
    Plots the given 2D field using matplotlib pcolormesh. Returns a matplotlib
    Axes object.

    Parameters
    ----------
    field: ndarray (2-D)
        2-D array of data to be plotted.
    ax: matplotlib Axes object, optional (default=None)
        An external Axes object that may be passed in to be further manipulated
        after calling this plotting function.
    size: int, optional (default=16)
        Sets the size of the Figure object.
    colorbar: bool, optional (default=False)
        Set to True to include a colorbar in the plot, False to not include a colorbar.
    y_min: int, optional (default=0)
        Lower limit of the y-axis to be plotted.
    y_max: int, optional (default=None)
        Upper limit of the y-axis to be plotted. If None, is set to the size of the
        y-axis for the input field.
    x_min: int, optional (default=0)
        Lower limit of the x-axis to be plotted.
    x_max: int, optional (default=None)
        Upper limit of the x-axis to be plotted. If None, is set to the size of the
        x-axis for the input field.
    title: str, optional (default=None)
        Title for the figure.
    title_size: int, optional (default=None)
        Sets the size for the figure title. If None, defaults to 2 times the 'size'
        parameter.
    axes_label_size: int, optional (default=None)
        Sets the size for the labels on the x and y axes. If None, defaults to
        1.8 times the 'size' parameter.
    axes: bool, optional (default=True)
        Determines whether to display the axes, including labels and ticks.
    ticks: bool, optional (default=None)
        Determines whether to display the axis tick marks and corresponding labels or not.
    xtick_spacing: int, optional (default=10)
        Sets the interval for ticks on along the x-axis.
    ytick_spacing: int, optional (default=10)
        Sets the interval for ticks along the y-axis.
    tick_size: int, optional (default=None)
        Sets the size of the tick labels for the axes. If None, defaults to the value
        of the 'size' parameter.
    cmap: matplotlib colormap, optional (default=plt.cm.Greys)
        Colormap used by pcolormesh for plotting the input field.
    edgecolors: valid matplotlib color, optional (default='black')
        Sets the color of the gird lines outlining the cells of the field.
        If set to 'none' does not display any grid lines.
    **anim_kwargs:
        Additional keyword arguments for the matplotlib animation.FuncAnimation()
        function.

    Returns
    -------
    ax: matplotlib Axes object
        Axes object that has either been passed in or created, then updated with
        this function.
    '''
    t,h,w = np.shape(data_fields)
    if y_max is None:
        y_max = h
    if x_max is None:
        x_max = w
    H = y_max - y_min
    W = x_max - x_min
    if ax is None:
        fig, ax = plt.subplots(figsize = (size, (H/W)*size))

    cropped_fields = data_fields[:, y_min:y_max, x_min:x_max]
    im = ax.pcolormesh(cropped_fields[0], cmap=cmap, edgecolors=edgecolors, **pcolor_kwargs)
    if invert_y:
        ax.invert_yaxis()

    if not axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    if colorbar:
        ax.colorbar()
    if title is not None:
        if title_size is None:
            title_size = 2*size
        ax.set_title(title, {'fontsize':title_size})
    if axes_label_size is None:
        axes_label_size = 1.8*size
    ax.set_ylabel('Y', {'fontsize':axes_label_size})
    ax.set_xlabel('X', {'fontsize':axes_label_size})

    # various code for tick control
    if ticks:
        ax.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
        if tick_size is None:
            tick_size = size
        x_labels = [str(int(label)+x_min+xtick_spacing) for label in ax.get_xticks().tolist()]
        y_labels = [str(int(label)+y_min+ytick_spacing) for label in ax.get_yticks().tolist()]
        x_ticks = [tick+0.5 for tick in ax.get_xticks()]
        y_ticks = [tick +0.5 for tick in ax.get_yticks()]
        if W % xtick_spacing == 0:
            x_trim = 2
        else:
            x_trim = 1
        if H % ytick_spacing == 0:
            y_trim = 2
        else:
            y_trim = 1
        ax.set_xticks(x_ticks[1:len(x_ticks)-x_trim], minor=True)
        ax.set_yticks(y_ticks[1:len(y_ticks)-y_trim], minor=True)
        ax.set_xticklabels(x_labels, fontsize=tick_size, minor=True)
        ax.set_yticklabels(y_labels, fontsize=tick_size, minor=True)
        ax.xaxis.set_major_formatter(plticker.NullFormatter())
        ax.yaxis.set_major_formatter(plticker.NullFormatter())
        ax.tick_params(
                axis='both',
                which='minor',
                direction='out',
                top='off',
                right='off',
                pad=8)
        ax.tick_params(
                axis='both',
                which='major',
                top='off',
                bottom='off',
                left='off',
                right='off')
    else:
        ax.tick_params(axis='both',
                        which='both',
                        bottom='off',
                        top='off',
                        left='off',
                        right='off',
                        labelleft='off',
                        labelbottom='off')

    plt.close() # prevents initial figure from being displayed in jupyter notebook

    def animate(i):
        im.set_array(cropped_fields[i].flatten())

    frames = t
    anim = animation.FuncAnimation(fig, animate, frames=frames, **anim_kwargs)

    return anim


###################################################################################################


def comparison_animate(data_fields, state_fields, size=20, 
                      y_min=0, y_max=None, x_min=0, x_max=None, title=None,
                      title_size=None, ticks=True, xtick_spacing=10, invert_y=True,
                      ytick_spacing=10, tick_size=None, field_cmap=plt.cm.Blues, state_cmap=plt.cm.rainbow,
                      edgecolors='none', anim_kwargs={}, **pcolor_kwargs):
    '''

    '''
    if data_fields.shape != state_fields.shape:
        raise ValueError("data_fields and state_fields must be same shape.")
    t,h,w = np.shape(data_fields)
    if y_max is None:
        y_max = h
    if x_max is None:
        x_max = w
    H = y_max - y_min
    W = x_max - x_min
    gridspec_kw = {'hspace':0.01}
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize = (size, (2*H/W)*size), gridspec_kw=gridspec_kw)

    cropped_fields = data_fields[:, y_min:y_max, x_min:x_max]
    im1 = ax1.pcolormesh(cropped_fields[0], cmap=field_cmap, edgecolors=edgecolors, **pcolor_kwargs)
    cropped_states = state_fields[:, y_min:y_max, x_min:x_max]
    im2 = ax2.pcolormesh(cropped_states[0], cmap=state_cmap, edgecolors=edgecolors, **pcolor_kwargs)
    if invert_y:
        ax1.invert_yaxis()
        ax2.invert_yaxis()

    if title is not None:
        if title_size is None:
            title_size = 2*size
        plt.set_title(title, {'fontsize':title_size})


    # various code for tick control
    if ticks:
        ax1.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
        ax1.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
        if tick_size is None:
            tick_size = size
        x_labels = [str(int(label)+x_min+xtick_spacing) for label in ax1.get_xticks().tolist()]
        y_labels = [str(int(label)+y_min+ytick_spacing) for label in ax1.get_yticks().tolist()]
        x_ticks = [tick+0.5 for tick in ax1.get_xticks()]
        y_ticks = [tick +0.5 for tick in ax1.get_yticks()]
        if W % xtick_spacing == 0:
            x_trim = 2
        else:
            x_trim = 1
        if H % ytick_spacing == 0:
            y_trim = 2
        else:
            y_trim = 1
        ax1.set_xticks(x_ticks[1:len(x_ticks)-x_trim], minor=True)
        ax1.set_yticks(y_ticks[1:len(y_ticks)-y_trim], minor=True)
        ax1.set_xticklabels(x_labels, fontsize=tick_size, minor=True)
        ax1.set_yticklabels(y_labels, fontsize=tick_size, minor=True)
        ax1.xaxis.set_major_formatter(plticker.NullFormatter())
        ax1.yaxis.set_major_formatter(plticker.NullFormatter())
        ax1.tick_params(
                axis='both',
                which='minor',
                direction='out',
                top='off',
                right='off',
                pad=8)
        ax1.tick_params(
                axis='both',
                which='major',
                top='off',
                bottom='off',
                left='off',
                right='off')
        
        ax2.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
        ax2.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
        if tick_size is None:
            tick_size = size
        x_labels = [str(int(label)+x_min+xtick_spacing) for label in ax2.get_xticks().tolist()]
        y_labels = [str(int(label)+y_min+ytick_spacing) for label in ax2.get_yticks().tolist()]
        x_ticks = [tick+0.5 for tick in ax2.get_xticks()]
        y_ticks = [tick +0.5 for tick in ax2.get_yticks()]
        if W % xtick_spacing == 0:
            x_trim = 2
        else:
            x_trim = 1
        if H % ytick_spacing == 0:
            y_trim = 2
        else:
            y_trim = 1
        ax2.set_xticks(x_ticks[1:len(x_ticks)-x_trim], minor=True)
        ax2.set_yticks(y_ticks[1:len(y_ticks)-y_trim], minor=True)
        ax2.set_xticklabels(x_labels, fontsize=tick_size, minor=True)
        ax2.set_yticklabels(y_labels, fontsize=tick_size, minor=True)
        ax2.xaxis.set_major_formatter(plticker.NullFormatter())
        ax2.yaxis.set_major_formatter(plticker.NullFormatter())
        ax2.tick_params(
                axis='both',
                which='minor',
                direction='out',
                top='off',
                right='off',
                pad=8)
        ax2.tick_params(
                axis='both',
                which='major',
                top='off',
                bottom='off',
                left='off',
                right='off')
        
    else:
        ax1.tick_params(axis='both',
                        which='both',
                        bottom='off',
                        top='off',
                        left='off',
                        right='off',
                        labelleft='off',
                        labelbottom='off')
        
        ax2.tick_params(axis='both',
                        which='both',
                        bottom='off',
                        top='off',
                        left='off',
                        right='off',
                        labelleft='off',
                        labelbottom='off')

    plt.close() # prevents initial figure from being displayed in jupyter notebook

    def animate(i):
        im1.set_array(cropped_fields[i].flatten())
        im2.set_array(cropped_states[i].flatten())

    frames = t
    anim = animation.FuncAnimation(fig, animate, frames=frames, **anim_kwargs)

    return anim