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

	docker build -t clcomp21-submission-sl:v0 .
	evalai push clcomp21-submission-sl:v0 -p cvpr21-sl-829

	# cleanup
	rm -r ./build/ || true

upload-rl:
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

	docker build -t clcomp21-submission-rl:v0 .
	evalai push clcomp21-submission-rl:v0 -p cvpr21-rl-829

	# cleanup
	rm -r ./build/ || true


upload-both:
	echo "TODO"

update:
	./update.sh


