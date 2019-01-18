# cambilight

https://www.youtube.com/watch?v=fWfalGsASjY&feature=youtu.be

PYTHONPATH=/tmp1/lifxlan/ python3 cv.py --config context.json --log-time

docker run -it --rm --device /dev/video0:/dev/video0 --network host -v $HOME:/tmp1 -v /media/yggdrasil/files:/tmp2 arthurgeron/docker-python3-opencv-security-camera bash


docker run -it --rm --network host -v `pwd`:/source arthurgeron/docker-python3-opencv-security-camera bash -c "cd source; ls; pip install -r requirements.test.txt; py.test"
