## Optical Pooled Screens

Analysis resources for the publication *Pooled genetic perturbation screens with image-based phenotypes*. This is an updated and expanded version of the repository for the original 2019 publication [*Optical pooled screens in human cells*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6886477/), available [here](https://github.com/feldman4/OpticalPooledScreens_2019).

### Installation (OSX)

Download the repository (e.g., on Github use the green "Clone or download" button, then "Download ZIP").

In Terminal, go to the NatureProtocols project directory and create a Python 3 virtual environment using a command like:

```bash
python3 -m venv venv
```

If the python3 command isn't available, you might need to specify the full path. E.g., if [Miniconda](https://conda.io/miniconda.html) is installed in the home directory:

```bash
~/miniconda3/bin/python -m venv venv
```

This creates a virtual environment called `venv` for project-specific resources. The commands in `install.sh` add required packages to the virtual environment:

```bash
sh install.sh
```

The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place.

Once installed, activate the virtual environment from the project directory:

```bash
source venv/bin/activate
```

Additionally, if using the CellPose segmentation method, this must be installed in the virtual environment:
```bash
pip install cellpose[gui]
```

## Running an example pipeline

To run the analysis pipeline with images from the [original publication](https://doi.org/10.1016/j.cell.2019.09.016), first download and install the [IBM Aspera Command Line Interface](https://www.ibm.com/support/knowledgecenter/SS4F2E_3.9/navigation/cli_welcome.html) (includes `ascp`) for interaction with the Cell-IDR databse. Then, set up an example directory:

```bash
python -m ops.paper.cell_idr setup_example example --ascp=<path/to/ascp/executable>
```

Note that if `ascp` is in your path, you can use `--ascp==ascp`.

Run the pipeline on the example data using [snakemake](https://snakemake.readthedocs.io/en/stable/) (after activating the virtual environment):

```bash
cd example
snakemake --cores all --configfile=config.yaml
```

## Additional example data

An example tile of 12-cycle SBS data is available in the original OpticalPooledScreens repository [here](https://github.com/feldman4/OpticalPooledScreens_2019/tree/master/example_data).

Additionally, all screening data presented in the [original publication](https://doi.org/10.1016/j.cell.2019.09.016) can be easily accessed from the public [Cell-IDR](https://idr.openmicroscopy.org/cell/) database (study `idr0071`) using `ascp` similar to above for the example pipeline:

```bash
python -m ops.paper.cell_idr get_cell_idr \
<destination> \
--experiment=<experiment> \
--well=<well> \
--tile=<tile> \
--ascp=<path/to/ascp/executable>
```
The following experiments are available:

| Cell-IDR experiment | Dataset |
|---------------------|---------|
| A | static p65-mNeonGreen screen in HeLa cells |
| B | static p65 antibody screen in HeLa cells |
| C | static p65 antibody screen in A549 cells |
| D | static p65 antibody screen in HCT-116 cells |
| E | Frameshift reporter screen in HeLa cells |
| F | Detection of combinatorial perturbations in HeLa cells |

Both `well` and `tile` can be set to `all` to download the full dataset. Figures 3 & 4 in the Nature Protocols manuscript were made using data from experiment C; functions for reproducing these figures are available [here](https://github.com/feldman4/NatureProtocols/tree/master/ops/paper).
