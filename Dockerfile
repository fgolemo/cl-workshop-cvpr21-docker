FROM fgolemo/cl21base:v1

RUN pip install --upgrade git+https://www.github.com/lebrice/Sequoia.git@cvpr_competition_dev#egg=sequoia[monsterkong]
WORKDIR /app
COPY ./build/ .
# just making sure we got everything
RUN pip install --upgrade -r submission/additional_pkgs.txt
CMD xvfb-run --server-args="-screen 0 800x600x24+32" python ./main.py --mode sl


# to build: docker build -t sequoia-seed .
# to run locally: docker run -it --rm sequoia-seed