
# Earthquake project

This project is designed to analyze earthquake and lunar data with the option of managing different graphs for analysis. For this we consider in the earthquake data latitude and longitude, magnitude, depth, time, periods of the year, groupings by magnitude. On the side of the moon data we take into consideration the data coming from ... and we use the fraction of illumination, declination, and distance (r/km)

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
Esta carpeta contiene los archivos que se utilizan y los path que se utilizan posteriormente en los scripts.

En `input/` tenemos el archivo de `earthquakes.xlsx` y `moon_data.xlsx`. En `output/` se contiene 
el archivo `minable.csv` que contiene la consolidación final de estas dos fuentes de información.

En `manage_file.py` es un script que maneja el path para luego leer y escribir sobre esos archivos.
En caso de querer usar nuevos archivos se recomienda dejarlos asignados con los nombres que actualmente se manejan,
pero en caso de cambiarlos se debe cambiar en este archivo el nombre

### 1.2 python_scripts

## 2.0 Variables

La data que se encuentra originalmente 
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
  


* Minable : es el consolidado de información. Contiene añadido PERIOD, MAG_SEG y las columnas
interpoladas.
  
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

## 3.0 Desarrollo