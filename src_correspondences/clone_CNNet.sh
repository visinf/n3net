git clone https://github.com/vcg-uvic/learned-correspondence-release.git CNNet
cd CNNet
cp ../*.py ./
cp ../archs/*.py ./archs/
mkdir logs
ln -s ../../../results_correspondences/pretrained_brown logs/ 
ln -s ../../../results_correspondences/pretrained_stpeters logs/ 