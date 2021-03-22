## Optical Pooled Screens

Analysis resources for the publication *Pooled genetic perturbation screens with image-based phenotypes*.

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

## Running example code

Once installed, activate the virtual environment from the project directory:

```bash
source venv/bin/activate
```

To run the analysis pipeline with an image from the [original publication](https://doi.org/10.1016/j.cell.2019.09.016), first download and install the [IBM Aspera Command Line Interface](https://www.ibm.com/support/knowledgecenter/SS4F2E_3.9/navigation/cli_welcome.html) (includes `ascp`) for interacting with the Cell-IDR databse. Then, set up an example directory:

```bash
python -m ops.paper.cell_idr setup_example example --ascp=<path/to/ascp/executable>
```

Run the pipeline on the example data using [snakemake](https://snakemake.readthedocs.io/en/stable/) (after activating the virtual environment):


```bash
cd example
snakemake --cores all -configfile=config.yaml
```