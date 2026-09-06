# Python / Jupyter Tutorials

Notebook versions of the course tutorials. Each notebook is meant to be run
cell by cell, with the narrative explanations and homework questions in the
markdown cells — the same structure as the MATLAB scripts in `../matlab/`.

## Setup

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## A note on committing notebooks

A `.ipynb` file is JSON that stores execution counts and every figure as
base64-encoded PNG data. Committed as-is, re-running a notebook produces a
large, unreadable diff, and a repository of plot-heavy tutorials grows very
quickly.

To avoid this, strip outputs automatically before each commit:

```bash
python3 -m pip install nbstripout
python3 -m nbstripout --install      # run once, inside this repository
```

Use the `python3 -m ...` form rather than the bare `pip` and `nbstripout`
commands: on macOS neither is reliably on `PATH`, and the module form works
regardless.

This installs a git filter, so notebooks are stored in git without outputs
while your working copy keeps them. Students clone a clean notebook and
generate the figures by running it, which is how the MATLAB tutorials
already work.

Check what got configured with:

```bash
git config --get filter.nbstripout.clean
```

If that prints an absolute interpreter path, the filter is pinned to one
specific Python install and will break if you ever remove or upgrade it.
Prefer the portable form, which follows whichever `python3` is on `PATH`:

```bash
git config filter.nbstripout.clean "python3 -m nbstripout"
git config diff.ipynb.textconv     "python3 -m nbstripout -t"
```

Either way the failure mode is safe: if the filter cannot run, git refuses
the commit rather than silently storing the outputs.
