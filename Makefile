#### TODO: FIX THIS TO WORK WITH CONDA ACTIVATE, see here https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
#### TODO: FOR NOW YOU HAVE TO MANUALLY CALL THE COMMANDS IN make install

# run with `make install EVALAI_TOKEN=...`
install:
	echo "=== CREATING NEW CONDA ENV 'clcomp21'"
	conda env create -f environment.yaml
	#conda activate clcomp21
	pip install -r requirements.txt
	evalai set_token EVALAI_TOKEN
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
	echo "=== RUNNING DOCKER BUILD AND UPLOAD"
	# prep copy of this project for container upload
	rm -r ./build/ || true
	mkdir build

	cp -r ./*.py ./build
	cp -r ./*.yaml ./build
	cp -r ./submission ./build

	rm -rf ./build/data || true
	rm -rf ./build/scripts || true
	rm -rf ./build/results || true

	docker build -t clcomp21-submission:v0 .
	#docker run -it sequoia-seed # this tests container locally
	evalai push clcomp21-submission:v0 -p cvpr21-dev-466 # replace this with link to SL challenge
	# cleanup
	rm -r ./build/ || true

upload-rl:
	echo "TODO"

upload-both:
	echo "TODO"

