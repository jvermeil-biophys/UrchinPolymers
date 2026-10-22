dir = "E:/WorkingData/LeicaData/26-03-18/";
list = getFileList(dir);

for (i = 0; i < list.length; i++) {
	if (endsWith(list[i], ".tif")) {
		run("TIFF Virtual Stack...", "open=" + dir + "/" + list[i]);
	}
}