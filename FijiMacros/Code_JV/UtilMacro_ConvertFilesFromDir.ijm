// dir = "D:/MagneticPincherData/Raw/23.09.06_Deptho/M2/";
sourceExtension = ".czi";

dir = getDirectory("Select a Directory")
list = getFileList(dir);

for (i = 0; i < list.length; i++) {
	if (endsWith(list[i], sourceExtension)) {
		open(dir + "/" + list[i]);
		selectImage("/" + list[i]);
		saveAs("Tiff", dir + "/" + substring(list[i], 0, list[i].length - 4));
	}
}

run("Close All");