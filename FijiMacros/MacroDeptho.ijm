dir = "D:/MagneticPincherData/Raw/23.09.06_Deptho/M1/";
list = getFileList(dir);

for (i = 0; i < list.length; i++) {
	open(dir + "/" + list[i]);
	selectImage(list[i]);
	run("In [+]");
	run("In [+]");
	ny = getHeight();
	nx = getWidth();
	makeRectangle(nx/2 - 74/2, ny/2 - 100/2, 74, 100);
	run("Threshold...");
	setAutoThreshold("Default dark no-reset stack");
	selectImage(list[i]);
	run("Analyze Particles...", "size=80-2000 circularity=0.70-1.00 display exclude clear include stack");
	saveAs("Results", dir + "/" + list[i][:-4] + "_Results.txt");
	
	// open("D:/MagneticPincherData/Raw/23.09.06_Deptho/M1/db1-1.tif");
	// selectImage("db1-1.tif");
}