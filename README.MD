# Adaptive Data Processing with [Sparrow](https://github.com/katanaml/sparrow)


The orignial License is GPL 3.0. This is a fork of the original project, which is available at [Sparrow](https://github.com/katanaml/sparrow)

## Adaption
- bug fixing
  - using llama-index-core==0.10.23.post1 to avoid the [unresolable import issue](https://github.com/katanaml/sparrow/pull/68#issue-2482213341)
- formating with ruff
  - install `pip install -r requirements-dev.txt`
  - `make format` or `ruff format`
- correcting type hints