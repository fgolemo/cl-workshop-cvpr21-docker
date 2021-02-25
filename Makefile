# run with `make install EVALAI_TOKEN=...`
install:
	./install.sh

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

	cp -r ./*.py ./buildı
	cp -r ./*.yaml ./build
	cp -r ./submission ./build

	rm -rf ./build/data || true
	rm -rf ./build/scripts || true
	rm -rf ./build/results || true

	docker build -t clcomp21-submission:v0 .
	# docker run -it clcomp21-submission:v0 # this tests container locally
	evalai push clcomp21-submission:v0 -p cvpr21-test-466 # replace this with link to SL challenge

	# cleanup
	rm -r ./build/ || true

upload-rl:
	echo "TODO"

upload-both:
	echo "TODO"

