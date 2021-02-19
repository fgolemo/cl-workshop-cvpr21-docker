install:
	echo "=== CREATING NEW CONDA ENV 'clcomp21'"
	conda env create -f environment.yaml
	conda activate clcomp21
	pip install -r requirements.txt
	echo "=== DONE. Now run 'make sl' for running the sl track or 'make rl' for the RL track or 'make help'"

sl:
	echo "=== RUNNING SL TRACK"
	python main.py --mode sl

rl:
	echo "=== RUNNING RL TRACK"
	python main.py --mode rl

both:
	echo "=== RUNNING BOTH TRACKS"
	python main.py --mode both

upload-sl:
	echo "TODO"

upload-rl:
	echo "TODO"

upload-both:
	echo "TODO"

