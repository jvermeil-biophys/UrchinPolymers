list = getList("image.titles");
i_stop = 500;

for (i = 0; i < list.length; i++) {
	imageName = list[i];
	N = imageName.length;
	selectImage(imageName);
	run("Duplicate...", "duplicate range=1-" + i_stop);
	duplicateName = substring(imageName, 0, N-4) + "-1.tif";
	selectImage(duplicateName);
	run("Invert", "stack");	
	run("Z Project...", "projection=Median");
	imageCalculator("Subtract create stack", duplicateName,"MED_" + duplicateName);
	resultName = "Result of " + imageName;
	
	// open(dir + "/" + list[i]);
		// run("In [+]");
	// ny = getHeight();
	// nx = getWidth();
	// makeRectangle(nx/2 - 74/2, ny/2 - 100/2, 74, 100);
	// run("Threshold...");
	// setAutoThreshold("Default dark no-reset stack");
	// selectImage(list[i]);
	// run("Analyze Particles...", "size=80-2000 circularity=0.70-1.00 display exclude clear include stack");
	// saveAs("Results", dir + "/" + list[i][:-4] + "_Results.txt");
	
	// open("D:/MagneticPincherData/Raw/23.09.06_Deptho/M1/db1-1.tif");
	// selectImage("db1-1.tif");
}
