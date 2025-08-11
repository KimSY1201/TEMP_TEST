# TEMP_TEST


pyinstaller 사용주의점
pyinstaller --hidden-import=scipy.ndimage --hidden-import=scipy.ndimage._ni_support --hidden-import=scipy.ndimage._nd_image --hidden-import=numpy.f2py main.py

해야함. 혹은 spec 파일 편집.