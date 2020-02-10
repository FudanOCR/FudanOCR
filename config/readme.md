# Config File
- `config.py`: Default configuration information, which records all parameters involved in the system.
- `example.py`: An example of how to call a parameter.
- `example.yaml`: Each model has its own yaml file. Change the different parameter values for each model in YAML file.

## others
There are two main paradigms:
- Configuration as local variable
- Configuration as a global singleton

The local variable route is recommended. We use it by default.