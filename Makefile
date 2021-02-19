#### TODO: FIX THIS TO WORK WITH CONDA ACTIVATE, see here https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate

install:
	echo "=== CREATING NEW CONDA ENV 'clcomp21'"
	conda env create -f environment.yaml
	conda activate clcomp21
	pip install -r requirements.txt
	echo "=== DONE. Now run 'make sl' for running the sl track or 'make rl' for the RL track or 'make help'"

sl:
	echo "=== RUNNING SL TRACK"
	python main.py --mode sl # this will take a bit

rl:
	echo "=== RUNNING RL TRACK"
	python main.py --mode rl # this will take a bit

both:
	echo "=== RUNNING BOTH TRACKS"
	python main.py --mode both # this will take a bit

upload-sl:
	echo "TODO"

upload-rl:
	echo "TODO"

upload-both:
	echo "TODO"

