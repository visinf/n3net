wget https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip -O Urban100.zip
unzip Urban100.zip -d Urban100
rm Urban100.zip

mkdir Urban100/rgb
find Urban100/image_SRF_4/ -name '*HR.png' | xargs -I file mv file Urban100/rgb/

rm -rf Urban100/image_SRF_4
rm -rf Urban100/image_SRF_2