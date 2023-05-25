
# Earthquake_moon project

This project is designed to analyze earthquake, solar and lunar gravitational data with the option of managing different graphs, histnorms, tables, trendlines for analysis. For this we consider in the earthquake data latitude and longitude, magnitude, depth, time, periods of the year, groupings by magnitude, groupings of earthquake clusters . On the side of the moon data we take into consideration the data coming from ... and we use the fraction of illumination, declination, and distance (r/km).

Robert Bostrom in Tectonic Consequences of the Earth’s Rotation notes that: 
> “ Von Helmholtz identified the species of motion (vortical) with respect to which there is no velocity potential in 1857. …, his insight has been disregarded by those of us struck by a tidal energy flux, astronomically well measured, that has seemed to “disappear” within the Earth. For scale, this is at least as great as that released in global seismicity. It may be desirable to examine the effects of actual external gravity, rather than those of a geocentric field that might well be attributed to Ptolemy. “

There is not a more perfect measurement of an “actual external gravity” than the phases of the moon. When the sun, the moon and the Earth are in syzygy , “actual external gravity” is at maximum and very well measured. It is unfortunate that 50 years of data amounts to less than 3 Earth/moon/sun perigean cycles each lasting 18.6 years. Contrast this to the well measured, almost 300 years of 25 solar cycles each lasting approximately 11 years.

Carlo Doglioni, notes in Polarized Plate tectonics, that it is an Earth’s scale phenomenology, and the energy necessary for its existence is not concentrated in limited zones (e.g., subduction zones), but it is contemporaneously distributed all over the whole Earth’s lithosphere, like the Earth’s rotation. Romashkova (2009) has recently shown how the planet seismicity indicates that the Earth’s lithosphere can be considered as a single whole. Only the global seismicity follows the Gutenberg-Richter law, while this simple relation does not hold when considering smaller portions of the Earth (Molchan, Kronrod, & Panza, 1997; Nekrasova & Kossobokov, 2006). The only mechanism that acts globally on the lithosphere and mantle is the Earth’s rotation.  
<!--AND THE MOON AND SUN-->

Based on the previous discussions, the observed phenomenology of plate tectonics confirms mantle convection, (there are waves on the ocean thus the ocean makes waves) but it appears to be governed from the top, i.e., an astronomically sheared lithosphere travels westward along the TE, facilitated by the decoupling allowed by the presence, at the top of the asthenosphere, of a continuous global flow within the Earth low-viscosity LVZ that is in superadiabatic conditions (Figure 86). 
Doglioni continues with, “The tidal forces are too small to generate earthquakes within continental areas, and for this reason they have been disregarded for long time.”  Continental Clusters 58, 40, 18, 9, 16, 35, 22, 39, 


## 1.0 Directory Project

```
📦 Earthquake_project
├─ df/
│  ├─ input/
│  │  ├─ earthquake.xlsx
│  │  └─ moon_data.xlsx
│  ├─ output/
│  │  └─ minable
│  └─ manage_file
└─ python_scripts
   ├─ main_run.ipynb
   ├─ NEW_EARTH_MOON_DATABASE.ipynb
   └─ functions.py
```

### 1.1 df
This folder contains the files that are used and the paths that are used later in the scripts.

In `input/` we have the `earthquakes.xlsx` and `moon_data.xlsx` files. In `output/` is contained
the `minable.csv` file that contains the final consolidation of these two sources of information.

In `manage_file.py` is a script that manages the path to then read and write to those files.
In case you want to use new files, it is recommended to leave them assigned with the names that are currently used,
but in case of changing them, the name must be changed in this file

### 1.2 python_scripts
This folder contains the python executables.

* `main_run.ipynb`: Jupyter notebook containing the main executables for testing and reviewing functionality. Runs the functions in functions.py, running data reading, filters, histograms, calculations, and others.

* `functions.py`: To simplify the amount of code that is handled from the main_run, this file is the container for functions that are executed.

* `new_database.ipynb`: In case of updating the earthquake and moon tables, this script must be run to generate the consolidated mineable again. It is important that the structure and column names are the same.

## 2.0 Variables
The variables contained in the input and output tables are the following
* Eartquake data
    
    | Variables |     |
    |-----------|-----|
    | `time`    | `id` |
    | `year`    | `updated`|
    | `month`   | `place` |
    | `day`   | `Po` |
    | `latitude`   |`Pais`|
    | `longitude`    | `type` |
    | `depth`    | `horizontalError`|
    | `mag`  | `depthError` |
    | `magType`   | `magError`|
    | `nst`   | `magNst` |
    | `gap`    | `status`|
    | `dmin` |`locationSource`|
    | `rms`   | `magSource` |
    | `net`   |  |



* Moon data
  
    | Variables |     |
    |-----------|-----|
    | `Year`    | `R/km` |
    | `Date`    | `DEC`|
    | `Month`   | `RA/h` |
    | `Day`   | `RA/°` |
    | `ill_frac`   | |
  


* Minable : is the consolidated information. Contains added PERIOD, MAG_SEG and columns
interpolated.
  
    | Variables |     |
    |-----------|-----|
    | `time`    | `NewMonth` |
    | `year`    | `cluster_label`|
    | `month`   | `acum_day` |
    | `day`   | `ill_frac` |
    | `latitude`   |`r/km`|
    | `longitude`    | `dec` |
    | `depth`    | `ra/h`|
    | `mag`  | `ra/°` |
    | `Pais`   | `ill_frac_interpolated`|
    | `PERIOD`   | `r/km_interpolated` |
    | `MAG_SEG`    | `dec_interpoalated`|
    | `NewDate` ||

## 3.0 Execution

