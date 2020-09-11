.. ############################################################################
.. README.rst
.. ==========
.. Author : Imaging of Things Group (ImoT)
.. ############################################################################

########
Pypeline
########
*Pypeline* is a `GPLv3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_ signal processing toolkit to
design and deploy holistic imaging pipelines.

This is a fork used for benchmarking and testing the code. 
This repository is also mirrored at:
https://c4science.ch/diffusion/10745/repository/master/

Installation
------------
For general installation, see ``doc/general/install.rst``

For installation on EPFL cluster, first you must install condo. You can run 

```
bash Miniconda3-latest-Linux-x86_64.sh 
```

And respond to the interactive instructions. Be sure to install conda into your home directory. 

Make a new directory which will contain pypeline and all of its packages. cd into this dir, then run:

```
git clone https://github.com/etolley/pypeline.git
cp pypeline/install.sh .
```

Afterwards, simply run:

```
source install.sh
```

For any future setup, run:

```
export PATH=$HOME/miniconda3/bin/:$PATH
cd pypeline/
source pypeline.sh --no_shell
```
