# Damping of Space Plasma
Plasma Exploration Project - Plasma Physics 525 Spring 2026

Presentaiton Slides: https://docs.google.com/presentation/d/1DHWIU1QQz4E1Ar3PmS_QcKLp3DKbotpf7I4ilr_cCEo/edit?usp=sharing


## Basis
This project is based on _A Prescription for the Turbulent Heating of Astrophysical Plasmas_ by Greg Howes(2010) which aims to calculate relative heating of ions and electrons due to kinetic dissipation. This will be done with Plasma in a Linear Uniform Magnetized Environment(PLUME). 

### Code
PLUME https://kgklein.github.io/PLUME/ is a code deveolepd by Howes and others to numericaly solve the Vlasov-Maxwell equation in hot magnetized plasmas. It can be used as a wrapper in windows but I was not able to figure that out so I ran it in a Linux virtual machine. The code saves the data points so edits to ploting do not require a fresh run.

###### *I realized once I had finished the project that I gave my virtual mahcine the minimum computing power so computing times are higher than what can easily be achinved

## Objective
Recreate figure 1 from Howes(2010) which is a contour plot of ion to electron heaing ratio on the ion to electron temperature vs ion plasma beta plane as seen below.

<img width="500"  alt="image" src="https://github.com/user-attachments/assets/4f0844c4-57fa-4e52-879a-5fbb0037c470" />

### Assumptions
Directly from Howes(2010) about cascade model
1) “the Kolmogorov hypothesis that the energy cascade is determined by local interactions (Kolmogorov 1941)”
2) “the turbulence maintains a state of critical balance at all scales (Goldreich & Sridhar 1995)”
3) “the linear kinetic damping rates are applicable in the non-linearly turbulent plasma”

Plasma properties
- The plasma is fully ionized, only protons and electrons that follow a Maxwellian distribution
- On the order of the Larmor radius, the mean magnetic field is a straight uniform field at     constant magnitude
- Turbulence is sub-Alfvenic and always satisfies critical balance so there are no direction biases at large scales
- Alfvenic cascade: MHD Alfven waves at large scales down to kinetic Alfven waves at small scales
  - transition at k⊥ρi ~ 1
- Non-relativistic 


## Plot 1 - Simplified Fit Function to Explain Physics

<img width="500" alt="image" src="https://github.com/user-attachments/assets/ae7e26e0-ba38-41c6-a37f-0eb6e06c4bd6" />

This fit fuction did not require PLUME and was give in Howes(2010). It was coded in Jupyter Notebook a does not requier heavy computing. Lots of the physics can be interpreted from this fit, but there qualitativle faliuer under Ti/Te ~ 10^-1 due to Te >> Ti not being realistic. Aside from that there is a jump at Ti/Te ~ 1 which is casued by chosing peicewise coeficients to better model the trasnision from Kinetic Theory to MHD. The integer contrours that are shown follow the physical competition between Landau Damping and Magneit Transit-Time Damping which casues bends to the right and left on either side of the plot. For beta_i > 1 the bend to the right is caused by a domination of ions which are ready to resonate with turbulace, which leads to their heating. For beta_i < 1 the bend to the right is caused by the domination of electroons which comes from ions losing their efficiney to resonate with turbulance relative to the electrons, which leads to a decrease in relative ion heating. Also for beta_i << 1 the scales get very tight because the energy cascade is getting to tiny scales.

## Plot 2 - Mostly Full Reproduciton

<img width="500" alt="image" src="https://github.com/user-attachments/assets/35292fe1-7ae1-4b38-bbba-079c9bf7feb2" />

The numerical solver goes through a 20x15 grid to get a total of 300 points. My run achinved 246/300 valid points over a ~30 min run. The most notable holes due to invalid points are in the low beta_i becasue of the tiny scale on teh cascade in the region.

## Plot 1 - Varying Paramaters - Anisotropy

The change I made was chaning from isotropy, apla_i = T_parallel / T_perp = 1, to a parallel aniosotropy, aplaph_1 < 1, and a perpendicular anisotropy, aplpha_i > 1. This will change the resonance condition between the waves and ions

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/7a96e286-13ca-4db1-9fa4-750100c90a5e" />

Solving time ~45 min for 3 plots x (10x15) grid for a total of 450 points. My run achinved 377/450 valid points.

#### Parallel Anisotropy aplah_i = 0.5 - Reduce ion heating efficieny due to new constraint

Firehose instability: At high beta_i and and parallel pressure the ions are not able to effectively absorb energy from turbulence so most the energy goes to the electrons.
- Lead to electron dominance
- More dark blue aka lower ion heating ratio


#### Perpendicular Anisotropy aplah_i = 2 - Enhances ion heating through new mechanisms

Stochastic heating: Non-resonant process that kicks ions to higher energies by small magnetic fluctuations.
Ion cyclotron resonance: Ions spin at same frequency as waves allowing them to directly absorb energy.
- Lead to ion dominance
- More orange aka higher ion heating ratio

Remaining Question:
Is that hole at beta_i ~ 1 from physics or a invalid point?

### Works Cited and Acknowledgements

Thank you to Vladimir Zhdankin for recomending the Howes(2010) paper which this project was based on.

Howes, G. G. (2010). A prescription for the turbulent heating of astrophysical plasmas. Monthly Notices of the Royal Astronomical Society: Letters, 409(1), L104–L108. 

Lynn B. Wilson III et al 2018 ApJS 236 41

Kristopher G. Klein et al 2025 Res. Notes AAS 9 102

PLUME_2025 K.G. Klein and G.G. Howes, Zenodo, v1.0.1 [Software]

Anthropic. (2024). Claude [Large language model]. (used for undertanding PLUME and help with coding)

Oracle. (2024). VirtualBox (Version 7.0) [Computer software].

Project Jupyter. (2024). JupyterLab [Computer software].

Python Software Foundation. (2024). Anaconda [Software distribution]. Anaconda, Inc.

