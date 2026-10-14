INSTALLATION INSTRUCTIONS

1. Install Python:
* Install miniforge3 from: https://github.com/conda-forge/miniforge
* > mamba init
* Use the Miniforge Prompt to create two new environments
	- spyder-env
	> mamba create --name spyder-env python=3.14
	> mamba activate spyder-env
	> mamba install numpy scipy pandas matplotlib

	- pyimagej-env
	> mamba create --name pyimagej-env python=3.14
	> mamba activate pyimagej-env
	> mamba install pyimagej openjdk=11
	> mamba install opencv
	> mamba install numpy scipy pandas matplotlib seaborn tifffile alphashape pillow scikit-image shapely json spyder-kernels

2. Install a stable version of Fiji
* Go to https://imagej.net/software/fiji/downloads
* Find the section : "Other downloads > Archive > Fiji-Stable.app" in the page
* Click on Fiji-Stable and download Fiji version: "20250514-1821/	2025-05-14 18:21"
* Unzip Fiji (as normal) and place it on your Desktop, rename the folder to 'Stable-Fiji' to avoid confusion with the version you will actually keep updated for other uses.

3. Initialize the code
* Run Spyder. You can find its location in your computer (C:\Users\<YOUR-USER-NAME>\miniforge3\envs\spyder-env\Scripts\spyder.exe) or maybe it will be in the start menu.
* Specify the Python interpreter: 
	- open Preferences (wrench icon), click on "Python interpreter" (3rd item in the left menu) 
	- chose: "Selected interpreter" and in the menu chose the version of Python within "pyimagej-env" (C:\Users\<YOUR-USER-NAME>\miniforge3\envs\pyimagej-env\python.exe).
	- Click on "ok" then start a new python console (close the existing one, a new one will restart). If you get a bug, you might have to reinstall a different version of spyder-kernels in the pyimagej-env environment. Do it with the Miniforge Prompt using:
	> mamba activate pyimagej-env
	> mamba install  spyder-kernels=<REQUIRED VERSION NUMBER>
* Open the main python file "AutoTrackmateToolbox.py"
	- There will be a section dedicated to the import of the imagej and scijava libraries. Do not forget to update the two paths:
Path_Fiji = "C:\\Users\\<YOUR-USER-NAME>\\Desktop\\Fiji-Stable.app\\"
Path_JAVA_HOME = "C:\\Users\\<YOUR-USER-NAME>\\miniforge3\\envs\\pyimagej-env\\Library\\lib\\jvm"
Manually check they point to existing and correct locations!
	- Run the two "Imports" cells (using CTRL+ENTER to run the code cell-by-cell). Check that all libraries are imported well (if there is not error returned, then everything's fine).
	- Imagej version should be "2.16.0/1.54p". If it shows "2.16.0/Inactive" close the console and redo the imports.

