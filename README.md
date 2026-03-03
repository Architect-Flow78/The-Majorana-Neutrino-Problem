# The-Majorana-Neutrino-Problem
# majorana_phase.py
# MAJORANA NEUTRINO -- TOROIDAL PHASE MATHEMATICS
# Framework: Matematica-tor
#
# PRIMARY:   theta in [0,1),  omega,  K
# SECONDARY: R=omega^(-2/3),  D=1-cos(Delta_theta),  mass, charge, spin
#
# Document sec.1: "1 = closed circle = full cycle"
#
# Dirac fermion:
#   full cycle = 1   =>  theta in [0, 1)   torus T^2
#   psi(theta+1) = psi(theta)
#   psi != psi*      particle != antiparticle
#   D(nu, nu_bar) in [0, 2]
#
# Majorana fermion:
#   full cycle = 1/2  =>  theta in [0, 0.5)   Mobius band
#   psi(theta+0.5) = psi(theta)
#   psi = psi*        particle == antiparticle
#   D(nu, nu_bar) = 0  always
#
# Phase potential (document: H = -sum K_ij cos(theta_i - theta_j)):
#   Dirac:    H_D = -K cos(2*pi*theta)      1 minimum per full cycle
#   Majorana: H_M = -K cos(4*pi*theta)      2 minima per full cycle
#             both minima physically identical => nu == nu_bar
#
# See-Saw in phase language:
#   Standard:  m_nu = m_D^2 / M_R
#   Phase:     omega_nu = omega_D^2 / omega_R
#   omega_R >> 1  =>  omega_nu << 1  =>  R_nu = omega_nu^(-2/3) >> 1
#   "near-zero mass" = delocalization, not a free parameter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------
# PHASE POTENTIAL
# Normalization: pi=1, full cycle=1, theta in [0,1)
# -----------------------------------------------------------------------

theta = np.linspace(0.0, 1.0, 1000)   # full cycle in [0,1)

# Dirac:    -cos(2*pi*theta)   one minimum at theta=0
# Majorana: -cos(4*pi*theta)   two minima at theta=0 and theta=0.5
V_D = -np.cos(2.0*np.pi*theta)
V_M = -np.cos(4.0*np.pi*theta)

idx_D = np.where(np.diff(np.sign(np.diff(V_D))) > 0)[0]+1
idx_M = np.where(np.diff(np.sign(np.diff(V_M))) > 0)[0]+1

print("Phase potential minima (theta in [0,1)):")
print(f"  Dirac:    {np.round(theta[idx_D],3)}  ({len(idx_D)} minimum)")
print(f"  Majorana: {np.round(theta[idx_M],3)}  ({len(idx_M)} minima)")

# -----------------------------------------------------------------------
# WAVE FUNCTIONS
# theta in [0,1) normalized units
#
# Dirac:    psi_D = exp(i*2*pi*theta/2) = exp(i*pi*theta)  complex
# Majorana: psi_M = cos(pi*theta)                          real, psi=psi*
# -----------------------------------------------------------------------

t_psi = np.linspace(0.0, 2.0, 1000)   # two full cycles
psi_D = np.exp(1j * np.pi * t_psi)
psi_M = np.cos(np.pi * t_psi)

# -----------------------------------------------------------------------
# PHASE DISTANCE  D_ij = 1 - cos(theta_i - theta_j)
# (document sec.4: emergence of distance)
#
# Antiparticle: theta_bar = -theta  (time-reversed winding)
#
# Dirac:    D(nu, nu_bar) = 1 - cos(2*pi*(theta-(-theta)))
#                         = 1 - cos(4*pi*theta)   mean = 1.0
# Majorana: nu == nu_bar  =>  D = 0 always
# -----------------------------------------------------------------------

D_dirac_anti = 1.0 - np.cos(4.0*np.pi*theta)
print(f"\nD(nu, nu_bar):")
print(f"  Dirac    mean = {np.mean(D_dirac_anti):.3f},  range [0,2]")
print("  Majorana D = 0 always")

# -----------------------------------------------------------------------
# MOBIUS BAND PARAMETRIZATION
# Encodes theta in [0, 0.5) geometrically:
#   x = (1 + s*cos(pi*u)) * cos(2*pi*u)
#   y = (1 + s*cos(pi*u)) * sin(2*pi*u)
#   z = s * sin(pi*u)
#   u in [0,1),  s in [-0.3, 0.3]
# After u: 0->1 the strip flips (s -> -s)
# => full geometric period = 2 loops = encodes half-period winding
# -----------------------------------------------------------------------

u_m = np.linspace(0.0, 1.0, 300)
s_m = np.linspace(-0.3, 0.3, 20)
U_m, S_m = np.meshgrid(u_m, s_m)
X_m = (1.0 + S_m*np.cos(np.pi*U_m)) * np.cos(2.0*np.pi*U_m)
Y_m = (1.0 + S_m*np.cos(np.pi*U_m)) * np.sin(2.0*np.pi*U_m)
Z_m = S_m * np.sin(np.pi*U_m)
ph_m = U_m % 1.0   # phase in [0,1)

# -----------------------------------------------------------------------
# SEE-SAW
# omega_nu = omega_D^2 / omega_R
# R_nu = omega_nu^(-2/3) = (omega_R / omega_D^2)^(2/3)
# -----------------------------------------------------------------------

omega_R  = np.logspace(0.0, 6.0, 200)
omega_D  = 1.0
omega_nu = omega_D**2 / omega_R
R_nu     = omega_nu**(-2.0/3.0)
R_D      = omega_D**(-2.0/3.0)

print(f"\nSee-Saw:")
print(f"  omega_R=1e3  =>  omega_nu={omega_D**2/1e3:.2e}  R_nu={( omega_D**2/1e3)**(-2/3):.1f}")
print(f"  omega_R=1e6  =>  omega_nu={omega_D**2/1e6:.2e}  R_nu={( omega_D**2/1e6)**(-2/3):.1f}")
print(f"  R_D = {R_D:.3f}")

# -----------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------

fig = plt.figure(figsize=(20,14))
fig.patch.set_facecolor('#0a0a1a')
fig.suptitle(
    'Majorana Neutrino -- Toroidal Phase Mathematics\n'
    'Document: "1 = full cycle"  |  Majorana: full cycle = 1/2  |  '
    'T^2 (torus) -> Mobius band  |  psi=psi*  |  D(nu,nu_bar)=0',
    fontsize=12, color='white', fontweight='bold')

# 1 -- Mobius band 3D
a1 = fig.add_subplot(3,3,1, projection='3d')
a1.set_facecolor('#0a0a1a')
a1.plot_surface(X_m, Y_m, Z_m, facecolors=plt.cm.hsv(ph_m), alpha=0.85, shade=False)
a1.set_title('Mobius Band\ntheta in [0, 0.5)  half-period winding\ncolor = phase in [0,1)',
             color='white', fontsize=8)
for p in [a1.xaxis.pane,a1.yaxis.pane,a1.zaxis.pane]: p.fill=False
a1.set_xticks([]); a1.set_yticks([]); a1.set_zticks([])

# 2 -- phase spaces polar
a2 = fig.add_subplot(3,3,2, projection='polar')
a2.set_facecolor('#0a0a1a')
th_full = np.linspace(0.0, 2.0*np.pi, 500)
th_half = np.linspace(0.0, np.pi, 300)
a2.fill(th_full, np.ones_like(th_full)*0.9, color='gold', alpha=0.2)
a2.plot(th_full, np.ones_like(th_full)*0.9, 'gold', lw=2, label='Dirac [0,1) full cycle')
a2.fill(th_half, np.ones_like(th_half)*0.6, color='cyan', alpha=0.3)
a2.plot(th_half, np.ones_like(th_half)*0.6, 'cyan', lw=2, label='Majorana [0,0.5) half')
a2.legend(fontsize=8, facecolor='#0a0a1a', labelcolor='white', loc='upper right')
a2.set_title('Phase spaces\nDirac: theta in [0,1)  Majorana: theta in [0,0.5)',
             color='white', fontsize=9)
a2.set_facecolor('#0a0a1a'); a2.tick_params(colors='white')

# 3 -- phase potentials
a3 = fig.add_subplot(3,3,3)
a3.set_facecolor('#0a0a1a')
a3.plot(theta, V_D, 'gold', lw=2.5, label='H_D = -K cos(2*pi*theta)   Dirac')
a3.plot(theta, V_M, 'cyan', lw=2.5, label='H_M = -K cos(4*pi*theta)   Majorana')
a3.scatter(theta[idx_D], V_D[idx_D], c='gold', s=100, zorder=5, label='Dirac min')
a3.scatter(theta[idx_M], V_M[idx_M], c='cyan', s=100, zorder=5, label='Majorana min')
a3.axhline(0, color='gray', lw=0.5, ls='--')
a3.set_xlabel('theta in [0,1)', color='white')
a3.set_ylabel('H(theta)', color='white')
a3.set_title('Phase potentials\nMajorana: doubled winding frequency, 2 identical minima',
             color='white', fontsize=9)
a3.legend(fontsize=7, facecolor='#0a0a1a', labelcolor='white')
a3.tick_params(colors='white')
for sp in a3.spines.values(): sp.set_edgecolor('gray')

# 4 -- wave functions
a4 = fig.add_subplot(3,3,4)
a4.set_facecolor('#0a0a1a')
a4.plot(t_psi, np.real(psi_D), 'gold', lw=2.0, label='Re[psi_Dirac]')
a4.plot(t_psi, np.imag(psi_D), 'gold', lw=1.5, ls='--', alpha=0.6, label='Im[psi_Dirac]')
a4.plot(t_psi, psi_M, 'cyan', lw=2.5, label='psi_Majorana = cos(pi*theta)  real!')
a4.axhline(0, color='gray', lw=0.5)
a4.axvline(1.0, color='white', lw=1, ls=':', alpha=0.5, label='one full cycle')
a4.axvline(2.0, color='white', lw=1, ls=':', alpha=0.3)
a4.set_xlabel('theta (cycles)', color='white')
a4.set_ylabel('psi(theta)', color='white')
a4.set_title('Wave functions\nMajorana: psi=psi*  Im vanishes identically',
             color='white', fontsize=9)
a4.legend(fontsize=7, facecolor='#0a0a1a', labelcolor='white')
a4.tick_params(colors='white')
for sp in a4.spines.values(): sp.set_edgecolor('gray')

# 5 -- D(nu, nu_bar)
a5 = fig.add_subplot(3,3,5)
a5.set_facecolor('#0a0a1a')
a5.fill_between(theta, D_dirac_anti, alpha=0.3, color='gold')
a5.plot(theta, D_dirac_anti, 'gold', lw=2.5,
        label='D(nu,nu_bar) Dirac = 1-cos(4*pi*theta)')
a5.axhline(0, color='cyan', lw=3,
           label='D(nu,nu_bar) Majorana = 0  always')
a5.set_xlabel('theta in [0,1)', color='white')
a5.set_ylabel('D = 1 - cos(Delta_theta)', color='white')
a5.set_title('Phase distance: nu vs nu_bar\nMajorana: D=0  particle==antiparticle',
             color='white', fontsize=9)
a5.legend(fontsize=8, facecolor='#0a0a1a', labelcolor='white')
a5.tick_params(colors='white')
for sp in a5.spines.values(): sp.set_edgecolor('gray')

# 6 -- See-Saw
a6 = fig.add_subplot(3,3,6)
a6.set_facecolor('#0a0a1a')
a6.loglog(omega_R, omega_nu, 'magenta', lw=2.5, label='omega_nu = omega_D^2 / omega_R')
a6.loglog(omega_R, R_nu/R_nu[0]*omega_nu[0], 'cyan', lw=1.5, ls='--',
          label='R_nu = omega_nu^(-2/3)  grows')
a6.axhline(1e-3, color='white', lw=1, ls=':', alpha=0.5, label='observed scale')
a6.set_xlabel('omega_R  heavy Majorana (primary)', color='white')
a6.set_ylabel('omega_nu  light neutrino', color='white')
a6.set_title('See-Saw: omega_nu = omega_D^2 / omega_R\nlarge omega_R => delocalized neutrino',
             color='white', fontsize=9)
a6.legend(fontsize=7, facecolor='#0a0a1a', labelcolor='white')
a6.tick_params(colors='white')
for sp in a6.spines.values(): sp.set_edgecolor('gray')

# 7 -- Mobius unrolled (phase map)
a7 = fig.add_subplot(3,3,7)
a7.set_facecolor('#0a0a1a')
uf = np.linspace(0,1,400); sf=np.linspace(-1,1,50)
UF,SF=np.meshgrid(uf,sf)
pf=(UF + SF*UF/2.0) % 1.0
a7.pcolormesh(UF, SF, pf, cmap='hsv', shading='auto')
a7.axhline(0, color='white', lw=2, ls='--', alpha=0.5)
a7.set_xlabel('theta (cycles along band)', color='white')
a7.set_ylabel('s (across band)', color='white')
a7.set_title('Mobius band unrolled\nPhase accumulates 0.5 twist per loop',
             color='white', fontsize=9)
a7.tick_params(colors='white')
for sp in a7.spines.values(): sp.set_edgecolor('gray')

# 8 -- summary
a8 = fig.add_subplot(3,3,8)
a8.set_facecolor('#0a0a1a'); a8.axis('off')
txt = ("MAJORANA NEUTRINO\n"
       "toroidal phase mathematics\n\n"
       "PRIMARY: theta, omega, K\n"
       "SECONDARY: mass, charge, spin\n\n"
       "Document sec.1:\n"
       "  1 = closed circle = full cycle\n"
       "  Majorana: full cycle = 1/2\n\n"
       "Dirac:\n"
       "  theta in [0,1)    torus T^2\n"
       "  psi != psi*\n"
       "  D(nu,nu_bar) in [0,2]\n"
       "  particle != antiparticle\n\n"
       "Majorana:\n"
       "  theta in [0,0.5)  Mobius\n"
       "  psi = psi*  real-valued\n"
       "  D(nu,nu_bar) = 0\n"
       "  particle == antiparticle\n\n"
       "H_D = -K cos(2*pi*theta)\n"
       "H_M = -K cos(4*pi*theta)\n"
       "Majorana: doubled frequency\n"
       "2 identical vacua per cycle\n\n"
       "See-Saw:\n"
       "  omega_nu=omega_D^2/omega_R\n"
       "  R=omega^(-2/3) -> large\n"
       "  delocalization not mass")
a8.text(0.03,0.98,txt, transform=a8.transAxes, color='white', fontsize=7.5,
        va='top', family='monospace',
        bbox=dict(boxstyle='round',facecolor='#0a0a1a',edgecolor='magenta',alpha=0.9))

# 9 -- Dirac vs Majorana trajectories
a9 = fig.add_subplot(3,3,9, projection='3d')
a9.set_facecolor('#0a0a1a')
tl = np.linspace(0.0, 2.0*np.pi, 200)
Rt, rt = 0.7, 0.25
# Dirac: winding number 1 per loop
a9.plot((Rt+rt*np.cos(tl*2))*np.cos(tl),
        (Rt+rt*np.cos(tl*2))*np.sin(tl),
        rt*np.sin(tl*2), 'gold', lw=2.5, label='Dirac  full cycle=1')
# Majorana: winding number 2 per loop (half period => twice the winding)
sm=0.15
a9.plot((1+sm*np.cos(tl/2))*np.cos(tl)*0.7,
        (1+sm*np.cos(tl/2))*np.sin(tl)*0.7,
        sm*np.sin(tl/2), 'cyan', lw=2.5, label='Majorana  full cycle=1/2')
a9.set_title('Torus (Dirac) vs Mobius (Majorana)\nDifferent topology => different particle type',
             color='white', fontsize=8)
a9.legend(fontsize=7, facecolor='#0a0a1a', labelcolor='white')
for p in [a9.xaxis.pane,a9.yaxis.pane,a9.zaxis.pane]: p.fill=False
a9.set_xticks([]); a9.set_yticks([]); a9.set_zticks([])

plt.tight_layout()
plt.savefig('majorana_phase.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.show()
print("Saved: majorana_phase.png")

# -----------------------------------------------------------------------
# ANIMATION -- torus morphing into Mobius band
# alpha=0: full cycle=1 (Dirac)
# alpha=1: full cycle=0.5 (Majorana)
# Im[psi] -> 0 as alpha -> 1
# -----------------------------------------------------------------------
print("Generating animation...")

fig_a = plt.figure(figsize=(14,7))
fig_a.patch.set_facecolor('#0a0a1a')
al = fig_a.add_subplot(1,2,1, projection='3d')
ar = fig_a.add_subplot(1,2,2, projection='3d')
for ax in [al,ar]:
    ax.set_facecolor('#0a0a1a')
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: p.fill=False
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

ttl = fig_a.suptitle('', color='gold', fontsize=11, fontweight='bold')
NF  = 60

def anim_mob(frame):
    al.cla(); ar.cla()
    for ax in [al,ar]:
        ax.set_facecolor('#0a0a1a')
        for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: p.fill=False
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    alpha = frame / float(NF)   # 0=torus, 1=Mobius

    # surface interpolation
    ua = np.linspace(0,1,300); sa=np.linspace(-0.25,0.25,15)
    UA,SA=np.meshgrid(ua,sa)
    twist = alpha * np.pi * UA
    Xa=(1+SA*np.cos(twist))*np.cos(2*np.pi*UA)*0.7
    Ya=(1+SA*np.cos(twist))*np.sin(2*np.pi*UA)*0.7
    Za=SA*np.sin(twist)
    pha=(UA + alpha*SA*UA/2.0)%1.0
    al.plot_surface(Xa,Ya,Za, facecolors=plt.cm.hsv(pha), alpha=0.8, shade=False)

    # particle point
    up = frame/float(NF)
    tw = alpha*np.pi*up
    xp=np.cos(2*np.pi*up)*0.7; yp=np.sin(2*np.pi*up)*0.7; zp=0.0
    al.scatter([xp],[yp],[zp],c='white',s=80,zorder=10)
    al.set_title(f'Topology: {(1-alpha)*100:.0f}% torus + {alpha*100:.0f}% Mobius\n'
                 f'full cycle = {1.0-alpha*0.5:.2f}',
                 color='white', fontsize=8)
    al.set_xlim(-1,1); al.set_ylim(-1,1); al.set_zlim(-0.4,0.4)

    # wave function
    th_w = np.linspace(0.0, 2.0, 500)
    # Dirac: exp(i*pi*theta), Majorana: cos(pi*theta)
    # interpolate Im -> 0
    psi_w = ((1-alpha)*np.exp(1j*np.pi*th_w) + alpha*np.cos(np.pi*th_w))
    ar.plot(np.cos(2*np.pi*th_w), np.sin(2*np.pi*th_w),
            np.real(psi_w), 'gold', lw=2, alpha=0.9, label='Re[psi]')
    ar.plot(np.cos(2*np.pi*th_w), np.sin(2*np.pi*th_w),
            np.imag(psi_w), 'cyan', lw=1.5, alpha=0.6, label='Im[psi]')
    ar.set_title('Wave function\nIm[psi] -> 0 as Dirac becomes Majorana',
                 color='white', fontsize=8)
    ar.set_xlim(-1.5,1.5); ar.set_ylim(-1.5,1.5); ar.set_zlim(-1.2,1.2)
    ar.legend(fontsize=7, facecolor='#0a0a1a', labelcolor='white')

    state = ("Majorana: psi=psi*" if alpha>0.9 else
             "Dirac: psi!=psi*"   if alpha<0.1 else
             f"transition alpha={alpha:.2f}")
    ttl.set_text(f'Torus -> Mobius  |  alpha={alpha:.2f}  |  {state}  |  '
                 f'D(nu,nu_bar)={2*alpha:.2f} -> 0')
    return []

an = animation.FuncAnimation(fig_a, anim_mob, frames=NF, interval=80, blit=False)
an.save('majorana_topology.gif', writer='pillow', fps=10, dpi=100,
        savefig_kwargs={'facecolor':'#0a0a1a'})
print("Saved: majorana_topology.gif")
plt.show()

print("\nRESULTS:")
print(f"  Dirac minima:    {np.round(theta[idx_D],3)} in [0,1)")
print(f"  Majorana minima: {np.round(theta[idx_M],3)} in [0,1)  => identical => nu==nu_bar")
print(f"  D(nu,nu_bar) Dirac mean={np.mean(D_dirac_anti):.3f},  Majorana=0")
print(f"  See-Saw: omega_R=1e6 => omega_nu={1e-6:.0e}, R_nu={1e4:.0f}")
