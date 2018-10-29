wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar -xf BSR_bsds500.tgz

mkdir BSDS500
mkdir BSDS500/data
mv ./BSR/BSDS500/data/images ./BSDS500/data/rgb

mkdir BSDS500/data/rgb/val68
xargs -I file -a BSDS_val68_list.txt cp BSDS500/data/rgb/val/file BSDS500/data/rgb/val68/

rm BSR_bsds500.tgz
rm -rf ./BSR