# svp-py
Simple python package for shortest vector problem

available algorithms:

- LLL
- enumeration
- IQP (gurobipy is required)

## Setup

```
git clone https://github.com/nariaki3551/svp-py.git
cd svp-py
python -m pip install requirements.txt
```

## Usage

```
usage: main.py [-h] [--num_randomize NUM_RANDOMIZE] instance {randomize,lll,enum,iqp} [{randomize,lll,enum,iqp} ...]

positional arguments:
  instance              Instance file of SVP Challenge
  {randomize,lll,enum,iqp}
                        Select algorithms from ['randomize', 'lll', 'enum', 'iqp']

options:
  -h, --help            show this help message and exit
```

## Example


- python main.py ./sample_mats/dim30seed0.txt lll 
- python main.py ./sample_mats/dim30seed0.txt lll enum
  - execute LLL and then enumeration
- python main.py ./sample_mats/dim30seed0.txt lll iqp
  - execute LLL and then integer quadratic programming search
