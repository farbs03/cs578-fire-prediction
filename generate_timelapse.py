import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from PIL import Image  # For GIF optimization
from datetime import datetime

def create_netcdf_timelapse(
    nc_file, 
    variable_name,
    output_gif='timelapse.gif',
    figsize=(10, 6),
    dpi=100,
    fps=5,
    cmap='viridis',
    vmin=None,
    vmax=None,
    title=None,
    coastline=True,
    borders=True,
    states=False,
    extent=None,
    optimize_gif=False
):
    """
    Create a timelapse GIF from a NetCDF variable.
    
    Parameters:
    -----------
    nc_file : str
        Path to NetCDF file
    variable_name : str
        Name of variable to animate
    output_gif : str, optional
        Output GIF filename
    figsize : tuple, optional
        Figure size (width, height)
    dpi : int, optional
        Figure DPI
    fps : int, optional
        Frames per second for animation
    cmap : str, optional
        Colormap name
    vmin/vmax : float, optional
        Color scale limits
    title : str, optional
        Plot title (if None, uses variable name)
    coastline/borders/states : bool, optional
        Whether to plot geographic features
    extent : list, optional
        [lon_min, lon_max, lat_min, lat_max] for map extent
    optimize_gif : bool, optional
        Whether to optimize GIF file size
    """
    
    # Load data
    ds = xr.open_dataset(nc_file)
    var = ds[variable_name]
    
    # Check time dimension
    if 'time' not in var.dims:
        raise ValueError("Variable must have 'time' dimension")
    
    # Get metadata
    long_name = var.attrs.get('long_name', variable_name)
    units = var.attrs.get('units', '')
    
    # Set up figure and projection
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent if provided
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add geographic features (only once, outside animation)
    if coastline:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    if states:
        ax.add_feature(cfeature.STATES, linewidth=0.3)
    
    # Initialize color normalization
    norm = Normalize(
        vmin=float(var.min()) if vmin is None else vmin,
        vmax=float(var.max()) if vmax is None else vmax
    )
    
    # Create initial plot (without colorbar)
    img = var.isel(time=0).plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )
    
    # Add colorbar (only once)
    cbar = fig.colorbar(img, ax=ax, pad=0.05)
    cbar.set_label(f"{long_name} {'[' + units + ']' if units != "None" else ''}")
    
    # Title with time formatting
    def format_time(time_val):
        """Convert numpy datetime to readable string"""
        dt = datetime.strptime(str(time_val)[:19], '%Y-%m-%dT%H:%M:%S')
        return dt.strftime('%Y-%m-%d')
    
    time_title = format_time(var['time'].isel(time=0).values)
    plt.title(f'{long_name}\n{time_title}')
    
    # Animation update function
    def update(frame):
        """Update the plot for each frame"""
        # Clear only the image data (not axes/features)
        img.set_array(var.isel(time=frame))
        
        # Update title with current time
        current_time = format_time(var['time'].isel(time=frame).values)
        ax.set_title(f'{long_name}\n{current_time}')
        
        return img
    
    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(var['time']),
        interval=1000/fps,
        blit=False  # Disable blitting to avoid issues
    )
    
    # Save GIF
    anim.save(output_gif, writer='pillow', fps=fps, dpi=dpi)
    
    plt.close()
    print(f"Timelapse saved to {output_gif}")
    return output_gif

if __name__ == "__main__":
    # Basic usage
    # create_netcdf_timelapse('fire_pred_dataset.nc', 'fire_mask', output_gif='fire_mask.gif')

    # With custom settings
    with xr.open_dataset("fire_pred_dataset.nc") as f:
        var_names = [i for i in f.data_vars]
    
    cmaps = {
        "fire_mask": "Reds",
        "frp": "YlOrRd",
        "lai_ave": "Greens",
        "everything_else": "viridis"
    }
    
    custom_vars = ['fire_mask', 'frp', 'lai_ave']
    
    for var in var_names:
        create_netcdf_timelapse(
            'fire_pred_dataset.nc',
            f'{var}',
            output_gif=f'./plots/{var}.gif',
            cmap=cmaps[var] if var in custom_vars else cmaps['everything_else'],
            fps=3,
            title=f'Los Angeles {var} in January 2025',
            coastline=True,
            borders=True,
            extent=[-119.09, -117.59, 34.02, 34.72]
        )