import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def set_axes_equal(ax):
    """
    Force 3D axes to have equal scale so spheres look like spheres.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = (x_limits[0] + x_limits[1]) / 2
    y_middle = (y_limits[0] + y_limits[1]) / 2
    z_middle = (z_limits[0] + z_limits[1]) / 2

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.invert_zaxis()
def plot_projective_curve_on_sphere(F_expr, *,
                                    sphere_res=180,
                                    band_eps=0.02,
                                    band_alpha=0.45,
                                    plane_z=1.0,
                                    plane_lim=3.0,
                                    plane_res=400,
                                    use_hemisphere=True):
    """
    F_expr: sympy expression in X,Y,Z (homogeneous polynomial recommended)
    sphere_res: sampling resolution on sphere (theta/phi grid)
    band_eps: thickness of |F|<eps band drawn on sphere (tune this!)
    plane_z: where to draw affine plane Z=plane_z (default 1)
    plane_lim: x,y range on plane for contour
    """

    # --- symbols and numeric functions ---
    X, Y, Z = sp.symbols('X Y Z', real=True)

    F = sp.lambdify((X, Y, Z), F_expr, 'numpy')
    f_aff_expr = sp.simplify(F_expr.subs({Z: 1}))
    f_aff = sp.lambdify((X, Y), f_aff_expr, 'numpy')  # here X,Y mean affine x,y

    # --- sphere grid (parameterization) ---
    # theta: azimuth [0,2pi], phi: polar [0,pi]
    theta = np.linspace(0, 2*np.pi, sphere_res)
    phi = np.linspace(0, np.pi, sphere_res//2 + 1)
    TH, PH = np.meshgrid(theta, phi)

    xs = np.sin(PH) * np.cos(TH)
    ys = np.sin(PH) * np.sin(TH)
    zs = np.cos(PH)

    # Option: only one representative of RP^2 (avoid antipodal duplication)
    if use_hemisphere:
        mask_hemi = zs >= 0
    else:
        mask_hemi = np.ones_like(zs, dtype=bool)

    # --- evaluate F on the sphere ---
    vals = F(xs, ys, zs)
    # normalize scale to make eps more stable across degrees
    scale = np.nanmax(np.abs(vals[mask_hemi]))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    vals_n = vals / scale

    # band on sphere: |F| < eps
    band = (np.abs(vals_n) < band_eps) & mask_hemi
    # --- build 3D figure ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 先設 box aspect（後面 set_xlim/ylim/zlim 後我會再設一次，最穩）
    ax.set_box_aspect((1, 1, 1))
    # 1) sphere wireframe
    ax.plot_wireframe(xs, ys, zs, rstride=6, cstride=6, linewidth=0.5)

    # 2) curve band on sphere (as translucent surface)
    # We plot only the band part of the sphere surface.
    xs_band = np.where(band, xs, np.nan)
    ys_band = np.where(band, ys, np.nan)
    zs_band = np.where(band, zs, np.nan)
    ax.plot_surface(xs_band, ys_band, zs_band, linewidth=0, antialiased=False, alpha=band_alpha)

    # 3) plane Z = plane_z with grid
    u = np.linspace(-plane_lim, plane_lim, 20)
    v = np.linspace(-plane_lim, plane_lim, 20)
    U, V = np.meshgrid(u, v)
    W = np.full_like(U, plane_z)
    ax.plot_wireframe(U, V, W, rstride=1, cstride=1, linewidth=0.3)

    # 4) affine curve f(x,y)=0 as a contour drawn on plane Z=plane_z
    gx = np.linspace(-plane_lim, plane_lim, plane_res)
    gy = np.linspace(-plane_lim, plane_lim, plane_res)
    GX, GY = np.meshgrid(gx, gy)
    # f_aff is F(x,y,1); if you set plane_z != 1, this is just a display plane.
    Fxy = f_aff(GX, GY)

    # contour in 2D, then lift to 3D by setting z=plane_z
        
    ax.contour(
        GX, GY, Fxy,
        levels=[0],
        zdir='z',
        offset=plane_z,
        linewidths=2
    )
    # 5) (optional but very “projective”): project sphere-band points to Z=1 via (x,y)=(X/Z,Y/Z)
    # This gives you the image of the projective curve on the affine chart.
    # Be careful near z=0.
    zmin = 0.15
    proj_mask = band & (zs > zmin)
    if np.any(proj_mask):
        xp = xs[proj_mask] / zs[proj_mask]
        yp = ys[proj_mask] / zs[proj_mask]
        ax.scatter(xp, yp, np.full_like(xp, plane_z), s=2)

    # cosmetics
    ax.set_xlim(-plane_lim, plane_lim)
    ax.set_ylim(-plane_lim, plane_lim)
    ax.set_zlim(-1.2, max(1.2, plane_z + 0.2))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=18, azim=35)

    ax.set_axis_off()
    plt.tight_layout()
    set_axes_equal(ax)
    plt.show()


if __name__ == "__main__":
    X, Y, Z = sp.symbols('X Y Z', real=True)

    # 例子：把這裡換成你的 homogeneous polynomial
    # 比如 Fermat cubic: X^3 + Y^3 - Z^3
    F_expr = X**3 + Y**3 - Z**3

    plot_projective_curve_on_sphere(
        F_expr,
        sphere_res=220,
        band_eps=0.015,
        plane_z=1.0,
        plane_lim=3.0,
        plane_res=600,
        use_hemisphere=True
    )
