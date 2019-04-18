@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%OPENDEEPFACESWAP_ROOT%\Sorter_FaceVector.py" "%WORKSPACE%\{0}\aligned" "{1}"

pause