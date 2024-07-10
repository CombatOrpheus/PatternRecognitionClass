find ../Notebooks -maxdepth 1 -name '*ipynb' | xargs -t -P2 jupyter nbconvert --to python
find ../Notebooks -maxdepth 1 -name '*py' -print0 | xargs -0 -t -I% mv % .
