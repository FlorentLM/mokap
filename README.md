# Installation

#### Miniconda (recommended):
* Clone this repository `git clone https://github.com/FlorentLM/mokap`
* Create environment `conda env create --file=environment.yml`
* That's it, you should be able to run it

# Usage

* Activate the conda environment `conda activate mokap`
* (Optional) Customise the example `main.py` file with your favourite text/code editor
* Run `./main.py`

# Troubleshooting

### Linux

    permission denied: ./main.py

**Fix**: make the file executable `chmod u+x ./main.py`

---

    Failed to open device xxxxx for XML file download. Error: 'The device cannot be operated on an USB 2.0 port. The device requires an USB 3.0 compatible port.'

**Fix**: Unplug and plug the camera(s) again

---

    Warning: Cannot change group of xxxx to 'video'.

**Fix**:  Add the local user to the *video* group: `sudo usermod -a -G video $LOGNAME`

---

    Error: 'Insufficient system resources exist to complete the API.'
    
or
    
    Too many open files. Reached open files limit

**Fix**:  Increase the number of file descriptors: `ulimit -n $(ulimit -H -n)`

_Note_: mokap normally does this automatically

---