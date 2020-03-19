""" FDTD visualization

This module supplies visualization methods for the FDTD Grid. They are
imported by the Grid class and hence are available as Grid methods.

"""

## Imports

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import numpy as np
# 2D visualization function


def visualize(
    grid,
    cmap="Blues",
    pbcolor="C3",
    pmlcolor=(0, 0, 0, 0.1),
    objcolor=(1, 0, 0, 0.1),
    srccolor="C0",
    detcolor="C2",
    show=True,
):
    """ visualize a projection of the grid and the optical energy inside the grid

    Args:
        grid: Grid: the grid instance to visualize

    Kwargs:
        cmap='Blues': the colormap to visualize the energy in the grid
        pbcolor='C3': the color to visualize the periodic boundaries
        pmlcolor=(0,0,0,0.1): the color to visualize the PML
        objcolor=(1,0,0,0.1): the color to visualize the objects in the grid
        objcolor='C0': the color to visualize the sources in the grid
    """
    # imports (placed here to circumvent circular imports)
    from .boundaries import _PeriodicBoundaryX, _PeriodicBoundaryY
    from .boundaries import (
        _PMLXlow,
        _PMLXhigh,
        _PMLYlow,
        _PMLYhigh,
    )

    #delete data last frame
    plt.clf()

    # just to create the right legend entries:
    plt.plot([], lw=7, color=objcolor, label="Objects")
    plt.plot([], lw=7, color=pmlcolor, label="PML")
    plt.plot([], lw=3, color=pbcolor, label="Periodic Boundaries")
    plt.plot([], lw=3, color=srccolor, label="Sources")
    plt.plot([], lw=3, color=detcolor, label="Detectors")

    # Grid energy
    grid_energy = np.sum(grid.E ** 2 + grid.H ** 2, -1)

    assert grid.Nx > 1 and grid.Ny > 1
    xlabel, ylabel = "x", "y"
    Nx, Ny = grid.Nx, grid.Ny
    pbx, pby = _PeriodicBoundaryX, _PeriodicBoundaryY
    pmlxl, pmlxh, pmlyl, pmlyh = _PMLXlow, _PMLXhigh, _PMLYlow, _PMLYhigh
    grid_energy = grid_energy[:, :]

    # visualize the energy in the grid
    plt.imshow(np.asarray(grid_energy), cmap=cmap, interpolation="sinc")

    # LineSource
    for source in grid.sources:
        _x = [source.x[0], source.x[-1]]
        _y = [source.y[0], source.y[-1]]

        plt.plot(_y, _x, lw=3, color=srccolor)

    # LineDetector
    for detector in grid.detectors:
        _x = [detector.x[0], detector.x[-1]]
        _y = [detector.y[0], detector.y[-1]]

        plt.plot(_y, _x, lw=3, color=detcolor)

    # Boundaries
    for boundary in grid.boundaries:
        if isinstance(boundary, pbx):
            _x = [-0.5, -0.5, float("nan"), Nx - 0.5, Nx - 0.5]
            _y = [-0.5, Ny - 0.5, float("nan"), -0.5, Ny - 0.5]
            plt.plot(_y, _x, color=pbcolor, linewidth=3)
        elif isinstance(boundary, pby):
            _x = [-0.5, Nx - 0.5, float("nan"), -0.5, Nx - 0.5]
            _y = [-0.5, -0.5, float("nan"), Ny - 0.5, Ny - 0.5]
            plt.plot(_y, _x, color=pbcolor, linewidth=3)
        elif isinstance(boundary, pmlyl):
            patch = ptc.Rectangle(
                xy=(-0.5, -0.5),
                width=boundary.thickness,
                height=Nx,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlxl):
            patch = ptc.Rectangle(
                xy=(-0.5, -0.5),
                width=Ny,
                height=boundary.thickness,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlyh):
            patch = ptc.Rectangle(
                xy=(Ny - 0.5 - boundary.thickness, -0.5),
                width=boundary.thickness,
                height=Nx,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlxh):
            patch = ptc.Rectangle(
                xy=(-0.5, Nx - boundary.thickness - 0.5),
                width=Ny,
                height=boundary.thickness,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)

    for obj in grid.objects:
        _x = (obj.x.start, obj.x.stop)
        _y = (obj.y.start, obj.y.stop)
            
        patch = ptc.Rectangle(
            xy=(min(_y) - 0.5, min(_x) - 0.5),
            width=max(_y) - min(_y),
            height=max(_x) - min(_x),
            linewidth=0,
            edgecolor="none",
            facecolor=objcolor,
        )
        plt.gca().add_patch(patch)

    # finalize the plot
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(Nx, -1)
    plt.xlim(-1, Ny)
    plt.figlegend()
    plt.tight_layout()
    if show:
        plt.show()
